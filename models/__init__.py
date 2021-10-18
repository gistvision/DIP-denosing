from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet
from .S2Snet import *
from .snet import *
import torch.nn as nn
import torchvision
import torch

class torchmodel(nn.Module):
    def __init__(self, model):
        super(torchmodel, self).__init__()
        self.s = nn.Sigmoid()
        self.model = model

    def forward(self, x):
        x_ = torch.cat([x,x])
        out = self.s(self.model(x_)['out'])
        out = torch.mean(out, dim = 0, keepdim=True)
        return out

def get_net(args):
    pad = 'reflection'
    input_depth = 1 if args.gray else 3
    if args.noisy_map:
        input_depth += 1
    n_channels = 1 if args.gray else 3
    NET_TYPE = args.net_type
    act_fun = args.act_func
    upsample_mode = 'bilinear'
    downsample_mode = 'stride'
    sigmoid = True

    if NET_TYPE == 'skip':
        skip_n33d = skip_n33u = args.hidden_layer
        skip_n11 = 4
        num_scales = 5
        print("[*] Net_type : skip with %d layer" % skip_n33d)
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=sigmoid, need_bias=True, pad=pad, act_fun=act_fun)
    elif NET_TYPE == 's2s':
        print("[*] Net_type : s2s")
        net = S2Snet(input_depth, n_channels, act_type=act_fun)
    elif NET_TYPE == 's2s_fixed':
        print("[*] Net_type : s2s_fixed")
        net = s2s_fixed(input_depth, n_channels, act_type= act_fun, bn_mode = args.bn_type)
    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios = [32, 16, 8, 4, 2, 1], fill_noise=False,pad=pad)
    elif NET_TYPE =='UNet':
        net = UNet(num_input_channels=input_depth, num_output_channels=3, 
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)
    elif NET_TYPE == 'dncnn':
        net = dncnn_s()
    elif NET_TYPE == 's2s96':
        net = S2Snet96(input_depth, n_channels, act_type= act_fun, bn_mode = args.bn_type)
    elif NET_TYPE == 's2sW':
        net = S2SnetW(input_depth, n_channels, act_type= act_fun)
    elif NET_TYPE == 's2sT':
        net = S2SnetT(input_depth, n_channels, act_type= act_fun)
    elif NET_TYPE == 's2s_':
        net = s2s_(input_depth, n_channels, act_type=act_fun)
    elif NET_TYPE == 's2s_g':
        net = s2s_g(input_depth, n_channels, act_type=act_fun)
    elif NET_TYPE == 's2s_4x':
        net = snet_4x(input_depth, n_channels, act_type= act_fun)
    elif NET_TYPE == 's2s_normal':
        net = s2s_normal(input_depth, n_channels, act_type=act_fun)
    elif NET_TYPE == 'S2SATnet1':
        net = S2SATnet1(input_depth, n_channels, act_type=act_fun)
    elif NET_TYPE == 'S2SATnet2':
        net = S2SATnet2(input_depth, n_channels, act_type= act_fun)


    elif NET_TYPE == 'Testnet':
        net = Testnet(input_depth, n_channels, act_type= act_fun)
    elif NET_TYPE == "MemNet":
        net = MemNet(input_depth, n_channels, 4, 6)
    else:
        assert False

    return net.type(args.dtype)
