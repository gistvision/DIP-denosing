import os
import argparse

def main_parser():
    parser = argparse.ArgumentParser()
    # shared param
    task_parsers = parser.add_subparsers(dest='task_type')
    parser.add_argument("--dip_type", default = "dip")
    parser.add_argument("--gray", action="store_true")

    # denoising param
    denoising_parser = task_parsers.add_parser('denoising')
    denoising_parser.add_argument("--eval_data", default="CSet9")
    denoising_parser.add_argument("--sigma", default=50, type=int)
    denoising_parser.add_argument("--lr", default=0.1, type=float)
    denoising_parser.add_argument('--reg_noise_std', default=1./20., type=float)

    # poisson param
    deblur_parser = task_parsers.add_parser('poisson')
    deblur_parser.add_argument("--eval_data", default="MNIST")
    deblur_parser.add_argument("--scale", default=0.1, type=float)
    deblur_parser.add_argument("--lr", default=0.1, type=float)
    deblur_parser.add_argument('--reg_noise_std', default=0.01, type=float)

    # network param
    parser.add_argument('--input_depth', default=3, type=int)
    parser.add_argument('--hidden_layer', default=64, type=int)
    parser.add_argument('--act_func', default="soft", type=str) # temporal experiment
    parser.add_argument("--optim", default="RAdam")
    parser.add_argument('--sigma_z', default=0.5, type=float)
    parser.add_argument("--net_type", default="s2s", type=str)

    # Additional methods.
    parser.add_argument("--force_steplr", action= "store_true")
    parser.add_argument("--extending", action="store_true")

    # BatchNorm methods.
    parser.add_argument("--bn_type", default="bn", type=str)
    parser.add_argument("--bn_fix_epoch", default=-1, type=int)

    # Extra_method related to DIP.
    parser.add_argument('--running_avg_ratio', default=0.99, type=float)

    # power of perturbation in divergence.
    parser.add_argument("--epsilon", default=0.5, type=float)# 1.6e-4
    parser.add_argument('--desc', default="", type=str)
    parser.add_argument("--exp_tag", default="", type=str)
    parser.add_argument('--show_every', default=500, type=int)
    parser.add_argument('--optim_init', default=0, type=int)
    parser.add_argument('--save_np', action='store_true')
    parser.add_argument('--epoch', default=0, type= int)
    parser.add_argument('--beta1', default=0.9, type=float) # Momentum.
    parser.add_argument('--beta2', default=0.999, type=float) # Adaptive learning rate.
    parser.add_argument('--noisy_map', action="store_true")
    parser.add_argument('--GT_noise', action="store_true")

    args = parser.parse_args()
    args.desc = "_" + args.desc

    return args
