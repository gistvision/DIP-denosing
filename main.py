import os
import argparse
import glob
import json

import cv2
import torch
import numpy as np
import pandas as pd

import loss
import models
import config_parser


from utils.common_utils import *
from utils.denoising_utils import *
from torch.utils.tensorboard import SummaryWriter

# beta version code
import additional_utils

def get_net(img_np, noise_np, args):
    net =  models.get_net(args)

    if args.dip_type in  ["dip_sure", "eSURE", "NCV_y", "eSURE_fixed", 'eSURE_new', 'eSURE_alpha', "eSURE_uniform", "eSURE_clip","eSURE_real", "no_div", "PURE", "PURE_dc", "dip_sure_new"]:
        net_input = cv2_to_torch(noise_np, dtype)
        print("[*] input_type : noisy image")
    else:
        INPUT = 'noise'
        input_depth = 1 if args.gray else 3
        # For SR, the get_noise should be same as img_np
        net_input = get_noise(input_depth, INPUT, (img_np.shape[1], img_np.shape[2])).type(dtype).detach()
        print("[*] input_type : noise")

    return net, net_input

def get_optim(name, net, lr, beta):
    if name == "adam":
        print("[*] optim_type : Adam")
        return torch.optim.Adam(net.parameters(), lr, beta)
    elif name == "adamw":
        print("[*] optim_type : AdamW (wd : 1e-2)")
        return torch.optim.AdamW(net.parameters(), lr, beta) # default weight decay is 1e-2.
    elif name == "RAdam":
        return additional_utils.RAdam(net.parameters(), lr, beta)
    else:
        raise NotImplementedError

def image_restorazation(file, args):
    # MAIN
    stat = {}
    task_type = args.task_type

    # Step 1. prepare clean & degradation(noisy) pair
    img_np, noisy_np = load_image_pair(file, task_type, args)
    if args.GT_noise:
        args.sigma = (img_np.astype(np.float) - noisy_np.astype(np.float)).std()
    # np_to_torch function from utils.common_utils.
    # _np : C,H,W [0, 255] -> _torch : C,H,W [0,1] scale
    img_torch   = cv2_to_torch(img_np, args.dtype)
    noise_torch = cv2_to_torch(noisy_np, args.dtype)

    # For PSNR measure.
    noisy_clip_np = np.clip(noisy_np, 0, 255)
    # Step 2. make model and model input
    net, net_input = get_net(img_np, noisy_np, args)
    net.train()

    # Step 3. set loss function.
    cal_loss = loss.get_loss(net, net_input, args)
    optimizer = get_optim(args.optim, net, args.lr, (args.beta1, args.beta2))
    if args.force_steplr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=.9, step_size=300)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 3000], gamma=0.5)

    # Step 4. optimization and inference.
    # Hyper_param for Learning
    psnr_noisy_last = 0
    psnr_gt_running = 0

    save_dir = args.save_dir

    # Ensemble methods.
    running_avg = None
    running_avg_ratio = args.running_avg_ratio

    image_name = file.split("/")[-1][:-4]
    np_save_dir = os.path.join(args.save_dir, image_name)
    os.makedirs(np_save_dir, exist_ok=True)

    stat["max_psnr"] = 0
    stat["max_ssim"] = 0
    stat["NUM_Backtracking"] = 0

    args.writer = SummaryWriter(log_dir="runs/%s/%s" % (args.exp_tag, args.desc + image_name))
    for ep in range(args.epoch):
        optimizer.zero_grad()
        total_loss, out = cal_loss(net_input, noise_torch)
        with torch.no_grad():
            mse_loss = torch.nn.functional.mse_loss(out, img_torch).item()
            diff_loss = total_loss.item() - mse_loss
            args.writer.add_scalar("loss/used_loss", total_loss.item(), global_step=ep)
            args.writer.add_scalar("loss/MSE_loss", mse_loss, global_step=ep)
            args.writer.add_scalar("loss/diff", diff_loss, global_step=ep)

        # _torch : C,H,W [0,1] scale => _np : C,H,W [0, 255]
        #out = torch_to_cv2(net(net_input))
        out = torch_to_cv2(out)
        psnr_noisy = calculate_psnr(noisy_clip_np, out)
        psnr_gt = calculate_psnr(img_np, out)
        lpips_noisy = calculate_lpips(noisy_clip_np, out, args.lpips)
        lpips_gt = calculate_lpips(img_np, out, args.lpips)
        args.writer.add_scalar("psnr/noisy_to_out", psnr_noisy, global_step=ep)
        args.writer.add_scalar("psnr/clean_to_out", psnr_gt, global_step=ep)
        args.writer.add_scalar("lpips/noisy_to_out", lpips_noisy, global_step=ep)
        args.writer.add_scalar("lpips/clean_to_out", lpips_gt, global_step=ep)

        if total_loss < 0:
            print('\nLoss is less than 0')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            break
        if (psnr_noisy - psnr_noisy_last < -5) and (ep > 5) :
            print('\nFalling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            stat["NUM_Backtracking"] += 1
            if stat["NUM_Backtracking"] > 10:
                break
            # continue
        else:
            # Running ensemble
            if True: #(ep % 50 == 0) and
                if running_avg is None:
                    running_avg = out
                else:
                    running_avg = running_avg * running_avg_ratio + out * (1 - running_avg_ratio)
                psnr_gt_running = calculate_psnr(img_np, running_avg)
                lpips_gt_running = calculate_lpips(img_np, running_avg, args.lpips, color="BGR")
                args.writer.add_scalar("psnr/clean_to_avg", psnr_gt_running, global_step=ep)
                args.writer.add_scalar("lpips/clean_to_avg", lpips_gt_running, global_step=ep)

                if (stat["max_psnr"] <= psnr_gt):
                    stat["max_step"] = ep
                    stat["max_psnr"] = psnr_gt
                    stat["max_psnr_avg"] = psnr_gt_running
                    stat["max_lpips_avg"] = lpips_gt_running
                    stat["max_lpips"] = lpips_gt 
                    max_out, maxavg_out = out.copy(),running_avg.copy()

                    #save file
                    if args.save_np:
                        state_dict = net.state_dict()
                        torch.save(state_dict, os.path.join(np_save_dir, "max_psnr_state_dict.pth"))

            if (ep == 200 or ep == 10) and (psnr_gt_running < psnr_gt):
                running_avg = None

            # args.writer.add_image("result/gt_noise_out_avg", np.concatenate([img_np, noisy_np, out, running_avg], axis=2), ep)
            print('Iteration %05d    total loss / MSE / diff %f / %f / %f   PSNR_noisy: %f   psnr_gt: %f PSNR_gt_sm: %f' % (
                    ep, total_loss.item(), mse_loss, diff_loss, psnr_noisy, psnr_gt, psnr_gt_running), end='\r')

            last_net = [x.detach().cpu() for x in net.parameters()]
            psnr_noisy_last=psnr_noisy
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if args.optim_init > 0:
            if ep % args.optim_init == 0:
                additional_utils.init_optim(net, optimizer)

    stat["final_ep"] = ep
    stat["final_psnr"] = psnr_gt
    stat["final_psnr_avg"] = psnr_gt_running
    stat["final_lpips_avg"]= lpips_gt_running
    stat["final_lpips"] = lpips_gt


    # Make final images
    if True:
        save_CHW_np(save_dir + "/%s.png" % (image_name), out)
        save_CHW_np(save_dir + "/%s_avg.png" % (image_name), running_avg)
        save_CHW_np(save_dir + "/%s_max.png" % (image_name), max_out)
        save_CHW_np(save_dir + "/%s_max_avg.png" % (image_name), maxavg_out)

        if args.gray:
            stat["final_ssim"] = calculate_ssim(img_np, out)
            stat["final_ssim_avg"] = calculate_ssim(img_np, running_avg)
            stat["max_ssim"] = calculate_ssim(img_np, max_out)
            stat["max_ssim_avg"] = calculate_ssim(img_np, maxavg_out)
        log_file = open(save_dir + "/%s_log.txt" % (image_name), "w")
        print(stat, file=log_file)
    print("%s psnr clean_out : %.2f, %.2f noise_out : %.2f, max %.2f, %.2f" % (
        image_name, psnr_gt_running, lpips_gt_running, psnr_noisy, stat["max_psnr"], stat["max_lpips"]), " " * 100)
    print(stat)
    args.writer.close()
    torch.cuda.empty_cache()
    return stat


def read_dataset_file_list(eval_data):
    dataset_dir = "./testset/%s/" % eval_data
    file_list1 = glob.glob(dataset_dir + "*.tif")
    file_list2 = glob.glob(dataset_dir + "*.png")
    file_list3 = glob.glob(dataset_dir + "*.JPG")
    file_list = file_list1 + file_list2 + file_list3
    return file_list


if __name__ == "__main__":
    # For REPRODUCIBILITY
    print("[*] reproduce mode On")
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dtype = torch.cuda.FloatTensor
        lpips = get_lpips("cuda")
    else:
        dtype = torch.FloatTensor
        lpips = get_lpips("cpu")
    args = config_parser.main_parser()
    args.save_dir = "./result/%s/%s/%s" % (args.task_type, args.exp_tag, args.dip_type + args.desc)
    os.makedirs(args.save_dir, exist_ok = True)

    # default epoch setup.
    if args.task_type == "denoising":
        args.epoch = 3000 if args.epoch == 0 else args.epoch
        args.save_point = [1, 10, 100, 500, 1000, 2000, 3000, 4000]
    elif args.task_type == "poisson":
        args.epoch = 3000 if args.epoch == 0 else args.epoch
        args.save_point = [1, 10, 100, 500, 1000, 2000, 3000, 4000]


    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.dtype = dtype
    args.lpips = lpips

    # file_list.
    file_list = read_dataset_file_list(args.eval_data)
    file_list = sorted(file_list)
    stat_list = []
    for file in file_list:
        print("[*] process image file : %s"  % file)
        stat = image_restorazation(file, args)
        stat_list.append(stat)

    data = pd.DataFrame(stat_list, index= [i.split("/")[-1] for i in file_list])
    os.makedirs("./csv/%s/%s/"  % (args.task_type, args.exp_tag), exist_ok=True)
    data.to_csv("./csv/%s/%s/%s.csv" % ( args.task_type, args.exp_tag ,args.dip_type+args.desc))
    print("experiment done")
    print(data)
