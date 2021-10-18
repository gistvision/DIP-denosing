import torch
import torch.nn as nn
from models.common import *
import math

class snet_test(nn.Module):
   def __init__(self, img_ch=3, out_ch=3, act_type="LeakyReLU", pad='reflection', bn_mode = "bn"):
        super(snet_test, self).__init__()
        self.conv1 = conv(img_ch, 64, kernel_size=5, stride=1, bias=False, pad=pad)
        self.conv2 = conv(64, 32, kernel_size=5, stride=1, bias=False, pad=pad)
        self.conv3 = conv(32, out_ch, kernel_size=5, stride=1, bias=False, pad=pad)
        self.conv_ = nn.Sequential(
            self.conv1, bn(64, bn_mode), act(act_type),
            bn(64, bn_mode), act(act_type), self.conv2,)

        self.conv__ = nn.Sequential(
            bn(32, bn_mode), act(act_type), self.conv3)
        self.Sig = nn.Sigmoid()


   def forward(self, x):
        x = (x - x.mean()) / (x.std())
        d0 = self.conv_(x)
        d0 = self.conv__(d0)
        return self.Sig(d0 / 30)

class snet_4x(nn.Module):
    def __init__(self, img_ch=3, out_ch=3, act_type="LeakyReLU", pad='reflection', bn_mode = "bn"):
        super(snet_4x, self).__init__()
        self.conv1 = conv(img_ch * 4, 64, kernel_size=5, stride=1, bias=False, pad=pad)
        self.conv2 = conv(64, 32, kernel_size=5, stride=1, bias=False, pad=pad)
        self.conv3 = conv(32, out_ch, kernel_size=5, stride=1, bias=False, pad=pad)
        self.conv_ = nn.Sequential(
            self.conv1, bn(64, bn_mode), act(act_type),
            bn(64, bn_mode), act(act_type), self.conv2,)

        self.conv__ = nn.Sequential(
            bn(32, bn_mode), act(act_type), self.conv3)
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        x = (x - x.mean()) / (x.std())
        x_bu = torch.flip(x, dims=[2])
        x_rl = torch.flip(x, dims=[3])
        x_all = torch.flip(x, dims=[2,3])
        x = torch.cat([x, x_bu, x_rl, x_all], dim=1)

        d0 = self.conv_(x)
        d0 = self.conv__(d0)
        return self.Sig(d0 / 30)



class S2SnetW(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection'):
        super(S2SnetW, self).__init__()
        self.net1 = snet_test(img_ch, output_ch, act_type, pad)
        self.net2 = snet_test(img_ch, output_ch, act_type, pad)

    def forward(self, x):
        return self.net2(self.net1(x))


class S2SnetT(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection'):
        super(S2SnetT, self).__init__()
        self.net = snet_test(img_ch, output_ch, act_type, pad)

    def forward(self, x):
        return self.net(self.net(x))

if __name__ == '__main__':
    x = torch.rand([1, 3, 481, 321])

    net = snet_test()
    print(net(x).shape)
