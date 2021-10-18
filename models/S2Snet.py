import torch
import torch.nn as nn
from models.common import *


class conv_block_sp(nn.Module):
    def __init__(self, ch_in, ch_out, down=False, act_fun='LeakyReLU', pad='reflection', group=1, bn_mode = "bn", bias=True):
        super(conv_block_sp, self).__init__()
        self.conv1 = conv(ch_in, ch_out, kernel_size=3, stride=1 if down is False else 2, bias=bias, pad=pad,
                          group=group)
        self.conv2 = conv(ch_out, ch_out, kernel_size=3, stride=1, bias=bias, pad=pad, group=group)
        self.conv = nn.Sequential(
            self.conv1, bn(ch_out, bn_mode if group == 1 else "groupNorm"), act(act_fun),
            self.conv2, bn(ch_out, bn_mode if group == 1 else "groupNorm"), act(act_fun))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_last(nn.Module):
    def __init__(self, ch_in, ch_out, down=False, act_fun='LeakyReLU', pad='reflection', group=1, bn_mode = "bn", bias=True):
        super(conv_block_last, self).__init__()
        self.conv1 = conv(ch_in, 64, kernel_size=3, stride=1 if down is False else 2, bias=bias, pad=pad, group=group)
        self.conv2 = conv(64, 32, kernel_size=3, stride=1, bias=bias, pad=pad, group=group)
        self.conv3 = conv(32, ch_out, kernel_size=3, stride=1, bias=bias, pad=pad, group=group)
        self.conv = nn.Sequential(
            self.conv1, bn(64, bn_mode if group == 1 else "groupNorm"), act(act_fun),
            self.conv2, bn(32, bn_mode if group == 1 else "groupNorm"), act(act_fun),
            self.conv3)

    def forward(self, x):
        x = self.conv(x)
        return x


class SIREN_layer(nn.Module):
    def __init__(self, ch_in, ch_out, frist = False, act_fun='sine', omega_0=30):
        super(SIREN_layer, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=True)
        self.act_fun = act(act_fun)
        self.omega_0 = omega_0
        self.in_features = ch_in
        self.frist = frist
        self.init()


    def init(self):
        with torch.no_grad():
            if self.frist:
                self.conv1.weight.uniform_(-1 / self.in_features,
                                           1 / self.in_features)
            else:
                self.conv1.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                                 np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        x = self.conv1(x)
        return self.act_fun(self.omega_0 * x)

class SIREN_CONV(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SIREN_CONV, self).__init__()
        self.conv1 = SIREN_layer(ch_in, 64, frist=True)
        self.conv2 = SIREN_layer(64, 32)
        self.conv3 = SIREN_layer(32, ch_out)
        self.conv = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3)

    def forward(self, x):
        x = self.conv(x)
        return x



class conv_block_skip(nn.Module):
    def __init__(self, ch_in, ch_out, act_fun='LeakyReLU', pad='reflection', group=1, bn_mode = "bn", bias=True):
        super(conv_block_skip, self).__init__()
        self.conv1 = conv(ch_in, ch_out, kernel_size=1, stride=1, bias=bias, pad=pad, group=group)
        self.conv = nn.Sequential(
            self.conv1, bn(ch_out, bn_mode if group == 1 else "groupNorm"), act(act_fun))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_concat(nn.Module):
    def __init__(self, ch_in, ch_out, act_fun='LeakyReLU', pad='reflection', group=1, bn_mode = "bn", bias=True):
        super(conv_block_concat, self).__init__()
        self.conv1 = conv(ch_in, ch_out, kernel_size=3, stride=1, bias=bias, pad=pad, group=group)
        self.conv2 = conv(ch_out, ch_out, kernel_size=3, stride=1, bias=bias, pad=pad, group=group)
        self.up = nn.Sequential(
            bn(ch_in),
            self.conv1, bn(ch_out, bn_mode if group == 1 else "groupNorm"), act(act_fun),
            self.conv2, bn(ch_out, bn_mode if group == 1 else "groupNorm"), act(act_fun))

    def forward(self, x):
        x = self.up(x)
        return x


class Concat_layer(nn.Module):
    def __init__(self, dim):
        super(Concat_layer, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class S2Snet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection'):
        super(S2Snet, self).__init__()
        enc_ch = [48, 48, 48, 48, 48]  # fixed
        dec_ch = [96, 96, 96, 96, 96]  # fixed
        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        self.Conv1 = conv_block_sp(ch_in=img_ch, ch_out=enc_ch[0], down=True, act_fun=act_type, pad=pad)  # h/2,  w/2
        self.Conv2 = conv_block_sp(ch_in=enc_ch[0], ch_out=enc_ch[1], down=True, act_fun=act_type, pad=pad)  # h/4,  w/4
        self.Conv3 = conv_block_sp(ch_in=enc_ch[1], ch_out=enc_ch[2], down=True, act_fun=act_type, pad=pad)  # h/8,  w/8
        self.Conv4 = conv_block_sp(ch_in=enc_ch[2], ch_out=enc_ch[3], down=True, act_fun=act_type,
                                   pad=pad)  # h/16, w/16
        self.Conv5 = conv_block_sp(ch_in=enc_ch[3], ch_out=enc_ch[4], down=True, act_fun=act_type,
                                   pad=pad)  # h/32, w/32

        self.conv = nn.Sequential(
            conv(enc_ch[4], dec_ch[4], kernel_size=3, stride=1, bias=True, pad=pad),
            bn(dec_ch[4]),
            act(act_type))

        self.Up_conv4 = conv_block_concat(ch_in=dec_ch[4] + enc_ch[3], ch_out=dec_ch[3], act_fun=act_type, pad=pad)
        self.Up_conv3 = conv_block_concat(ch_in=dec_ch[3] + enc_ch[2], ch_out=dec_ch[2], act_fun=act_type, pad=pad)
        self.Up_conv2 = conv_block_concat(ch_in=dec_ch[2] + enc_ch[1], ch_out=dec_ch[1], act_fun=act_type, pad=pad)
        self.Up_conv1 = conv_block_concat(ch_in=dec_ch[1] + enc_ch[0], ch_out=dec_ch[0], act_fun=act_type, pad=pad)
        self.Conv_last1 = conv_block_last(dec_ch[0] + img_ch, output_ch, act_fun=act_type, pad=pad)  # concat
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # h/2 w/2
        x2 = self.Conv2(x1)  # h/4 w/4
        x3 = self.Conv3(x2)  # h/8 w/8
        x4 = self.Conv4(x3)  # h/16 w/16
        x5 = self.Conv5(x4)  # h/32 w/32

        x6 = self.conv(x5)

        d5 = self.upsample(x6)  # h/16 w/16
        d5 = self.concat([d5, x4])
        d4 = self.Up_conv4(d5)

        d4 = self.upsample(d4)  # h/8  w/8
        d4 = self.concat([d4, x3])
        d3 = self.Up_conv3(d4)

        d3 = self.upsample(d3)  # h/4  w/4
        d3 = self.concat([d3, x2])
        d2 = self.Up_conv2(d3)

        d2 = self.upsample(d2)  # h/2  w/2
        d2 = self.concat([d2, x1])
        d1 = self.Up_conv1(d2)

        d0 = self.upsample(d1)  # h    w
        d0 = self.concat([d0, x])
        d0 = self.Conv_last1(d0)
        return self.Sig(d0)


class s2s_(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection', bn_mode = "bn"):
        super(s2s_, self).__init__()
        enc_ch = [48, 48, 48, 48, 48]  # fixed
        dec_ch = [96, 96, 96, 96, 96]  # fixed
        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        self.Conv_first1 = conv_block_last(img_ch, enc_ch[0], act_fun=act_type, pad=pad, bn_mode=bn_mode, bias=False)  # concat
        self.Conv1 = conv_block_sp(ch_in=img_ch + enc_ch[0], ch_out=enc_ch[0], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode, bias=False)  # h/2,  w/2
        self.Conv2 = conv_block_sp(ch_in=enc_ch[0], ch_out=enc_ch[1], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode, bias=False)  # h/4,  w/4
        self.Conv3 = conv_block_sp(ch_in=enc_ch[1], ch_out=enc_ch[2], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode, bias=False)  # h/8,  w/8
        self.Conv4 = conv_block_sp(ch_in=enc_ch[2], ch_out=enc_ch[3], down=True, act_fun=act_type,
                                   pad=pad,  bn_mode=bn_mode)  # h/16, w/16
        self.Conv5 = conv_block_sp(ch_in=enc_ch[3], ch_out=enc_ch[4], down=True, act_fun=act_type,
                                   pad=pad,  bn_mode=bn_mode)  # h/32, w/32

        self.conv = nn.Sequential(
            conv(enc_ch[4], dec_ch[4], kernel_size=3, stride=1, bias=False, pad=pad),
            bn(dec_ch[4], bn_mode),
            act(act_type))

        self.Up_conv4 = conv_block_concat(ch_in=dec_ch[4] + enc_ch[3], ch_out=dec_ch[3], act_fun=act_type, pad=pad, bn_mode=bn_mode, bias=False)
        self.Up_conv3 = conv_block_concat(ch_in=dec_ch[3] + enc_ch[2], ch_out=dec_ch[2], act_fun=act_type, pad=pad, bn_mode=bn_mode, bias=False)
        self.Up_conv2 = conv_block_concat(ch_in=dec_ch[2] + enc_ch[1], ch_out=dec_ch[1], act_fun=act_type, pad=pad, bn_mode=bn_mode, bias=False)
        self.Up_conv1 = conv_block_concat(ch_in=dec_ch[1] + enc_ch[0], ch_out=dec_ch[0], act_fun=act_type, pad=pad, bn_mode=bn_mode, bias=False)
        self.Conv_last1 = conv_block_last(dec_ch[0] + img_ch, output_ch, act_fun=act_type, pad=pad, bn_mode=bn_mode, bias=False)  # concat

        self.Sig = nn.Sigmoid()
        if img_ch == 4:
            self.mean = torch.tensor([0.406, 0.456, 0.485, 0]).view(1, 4, 1, 1)
            self.std  = torch.tensor([0.225, 0.224, 0.229, 1]).view(1, 4, 1, 1)
        if img_ch == 3:
            self.mean = torch.tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1)
            self.std  = torch.tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.449])
            self.std = torch.tensor([0.226])

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        # encoding path
        x0 = self.Conv_first1(x)
        x0 = self.concat([x0, x])
        x1 = self.Conv1(x0)  # h/2 w/2
        x2 = self.Conv2(x1)  # h/4 w/4
        x3 = self.Conv3(x2)  # h/8 w/8
        x4 = self.Conv4(x3)  # h/16 w/16
        x5 = self.Conv5(x4)  # h/32 w/32

        x6 = self.conv(x5)

        d5 = self.upsample(x6)  # h/16 w/16
        d5 = self.concat([d5, x4])
        d4 = self.Up_conv4(d5)

        d4 = self.upsample(d4)  # h/8  w/8
        d4 = self.concat([d4, x3])
        d3 = self.Up_conv3(d4)

        d3 = self.upsample(d3)  # h/4  w/4
        d3 = self.concat([d3, x2])
        d2 = self.Up_conv2(d3)

        d2 = self.upsample(d2)  # h/2  w/2
        d2 = self.concat([d2, x1])
        d1 = self.Up_conv1(d2)

        d0 = self.upsample(d1)  # h    w
        d0 = self.concat([d0, x])
        d0 = self.Conv_last1(d0)
        return self.Sig(d0)

class s2s_fixed(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection', bn_mode = "bn"):
        super(s2s_fixed, self).__init__()
        enc_ch = [48, 48, 48, 48, 48]  # fixed
        dec_ch = [96, 96, 96, 96, 96]  # fixed
        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        self.Conv1 = conv_block_sp(ch_in=img_ch, ch_out=enc_ch[0], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode)  # h/2,  w/2
        self.Conv2 = conv_block_sp(ch_in=enc_ch[0], ch_out=enc_ch[1], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode)  # h/4,  w/4
        self.Conv3 = conv_block_sp(ch_in=enc_ch[1], ch_out=enc_ch[2], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode)  # h/8,  w/8
        self.Conv4 = conv_block_sp(ch_in=enc_ch[2], ch_out=enc_ch[3], down=True, act_fun=act_type,
                                   pad=pad,  bn_mode=bn_mode)  # h/16, w/16
        self.Conv5 = conv_block_sp(ch_in=enc_ch[3], ch_out=enc_ch[4], down=True, act_fun=act_type,
                                   pad=pad,  bn_mode=bn_mode)  # h/32, w/32

        self.conv = nn.Sequential(
            conv(enc_ch[4], dec_ch[4], kernel_size=3, stride=1, bias=True, pad=pad),
            bn(dec_ch[4], bn_mode),
            act(act_type))

        self.Up_conv4 = conv_block_concat(ch_in=dec_ch[4] + enc_ch[3], ch_out=dec_ch[3], act_fun=act_type, pad=pad, bn_mode=bn_mode)
        self.Up_conv3 = conv_block_concat(ch_in=dec_ch[3] + enc_ch[2], ch_out=dec_ch[2], act_fun=act_type, pad=pad, bn_mode=bn_mode)
        self.Up_conv2 = conv_block_concat(ch_in=dec_ch[2] + enc_ch[1], ch_out=dec_ch[1], act_fun=act_type, pad=pad, bn_mode=bn_mode)
        self.Up_conv1 = conv_block_concat(ch_in=dec_ch[1] + enc_ch[0], ch_out=dec_ch[0], act_fun=act_type, pad=pad, bn_mode=bn_mode)
        self.Conv_last1 = conv_block_last(dec_ch[0] + img_ch, output_ch, act_fun=act_type, pad=pad, bn_mode=bn_mode)  # concat
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # h/2 w/2
        x2 = self.Conv2(x1)  # h/4 w/4
        x3 = self.Conv3(x2)  # h/8 w/8
        x4 = self.Conv4(x3)  # h/16 w/16
        x5 = self.Conv5(x4)  # h/32 w/32

        x6 = self.conv(x5)

        d5 = self.upsample(x6)  # h/16 w/16
        d5 = self.concat([d5, x4])
        d4 = self.Up_conv4(d5)

        d4 = self.upsample(d4)  # h/8  w/8
        d4 = self.concat([d4, x3])
        d3 = self.Up_conv3(d4)

        d3 = self.upsample(d3)  # h/4  w/4
        d3 = self.concat([d3, x2])
        d2 = self.Up_conv2(d3)

        d2 = self.upsample(d2)  # h/2  w/2
        d2 = self.concat([d2, x1])
        d1 = self.Up_conv1(d2)

        d0 = self.upsample(d1)  # h    w
        d0 = self.concat([d0, x])
        d0 = self.Conv_last1(d0)
        return self.Sig(d0)
    
class s2s_normal(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection', bn_mode = "bn"):
        super(s2s_normal, self).__init__()
        enc_ch = [48, 48, 48, 48, 48]  # fixed
        dec_ch = [96, 96, 96, 96, 96]  # fixed
        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        self.Conv1 = conv_block_sp(ch_in=img_ch, ch_out=enc_ch[0], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode)  # h/2,  w/2
        self.Conv2 = conv_block_sp(ch_in=enc_ch[0], ch_out=enc_ch[1], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode)  # h/4,  w/4
        self.Conv3 = conv_block_sp(ch_in=enc_ch[1], ch_out=enc_ch[2], down=True, act_fun=act_type, pad=pad,  bn_mode=bn_mode)  # h/8,  w/8
        self.Conv4 = conv_block_sp(ch_in=enc_ch[2], ch_out=enc_ch[3], down=True, act_fun=act_type,
                                   pad=pad,  bn_mode=bn_mode)  # h/16, w/16
        self.Conv5 = conv_block_sp(ch_in=enc_ch[3], ch_out=enc_ch[4], down=True, act_fun=act_type,
                                   pad=pad,  bn_mode=bn_mode)  # h/32, w/32

        self.conv = nn.Sequential(
            conv(enc_ch[4], dec_ch[4], kernel_size=3, stride=1, bias=True, pad=pad),
            bn(dec_ch[4], bn_mode),
            act(act_type))

        self.Up_conv4 = conv_block_concat(ch_in=dec_ch[4] + enc_ch[3], ch_out=dec_ch[3], act_fun=act_type, pad=pad, bn_mode=bn_mode)
        self.Up_conv3 = conv_block_concat(ch_in=dec_ch[3] + enc_ch[2], ch_out=dec_ch[2], act_fun=act_type, pad=pad, bn_mode=bn_mode)
        self.Up_conv2 = conv_block_concat(ch_in=dec_ch[2] + enc_ch[1], ch_out=dec_ch[1], act_fun=act_type, pad=pad, bn_mode=bn_mode)
        self.Up_conv1 = conv_block_concat(ch_in=dec_ch[1] + enc_ch[0], ch_out=dec_ch[0], act_fun=act_type, pad=pad, bn_mode=bn_mode)
        self.Conv_last1 = conv_block_last(dec_ch[0] + img_ch, output_ch, act_fun=act_type, pad=pad, bn_mode=bn_mode)  # concat
        self.Sig = nn.Sigmoid()
        if img_ch == 4:
            self.mean = torch.tensor([0.406, 0.456, 0.485, 0]).view(1, 4, 1, 1)
            self.std  = torch.tensor([0.225, 0.224, 0.229, 1]).view(1, 4, 1, 1)
        if img_ch == 3:
            self.mean = torch.tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1)
            self.std  = torch.tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.449])
            self.std = torch.tensor([0.226])

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        # encoding path
        x1 = self.Conv1(x)  # h/2 w/2
        x2 = self.Conv2(x1)  # h/4 w/4
        x3 = self.Conv3(x2)  # h/8 w/8
        x4 = self.Conv4(x3)  # h/16 w/16
        x5 = self.Conv5(x4)  # h/32 w/32

        x6 = self.conv(x5)

        d5 = self.upsample(x6)  # h/16 w/16
        d5 = self.concat([d5, x4])
        d4 = self.Up_conv4(d5)

        d4 = self.upsample(d4)  # h/8  w/8
        d4 = self.concat([d4, x3])
        d3 = self.Up_conv3(d4)

        d3 = self.upsample(d3)  # h/4  w/4
        d3 = self.concat([d3, x2])
        d2 = self.Up_conv2(d3)

        d2 = self.upsample(d2)  # h/2  w/2
        d2 = self.concat([d2, x1])
        d1 = self.Up_conv1(d2)

        d0 = self.upsample(d1)  # h    w
        d0 = self.concat([d0, x])
        d0 = self.Conv_last1(d0)
        return self.Sig(d0)

class s2s_g(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection', bn_mode = "groupNorm"):
        super(s2s_g, self).__init__()
        enc_ch = [24, 24, 24, 24, 24]  # fixed
        dec_ch = [24, 24, 24, 24, 24]  # fixed
        self.img_ch = img_ch
        self.upsample = nn.Upsample(scale_factor=2)
        self.concat = Concat_layer(1)
        self.Conv1 = conv_block_sp(ch_in=img_ch *4, ch_out=enc_ch[0], down=True, act_fun=act_type, pad=pad, group=4,
                                   bn_mode=bn_mode)  # h/2,  w/2
        self.Conv2 = conv_block_sp(ch_in=enc_ch[0], ch_out=enc_ch[1], down=True, act_fun=act_type, pad=pad, group=4,
                                   bn_mode=bn_mode)  # h/4,  w/4
        self.Conv3 = conv_block_sp(ch_in=enc_ch[1], ch_out=enc_ch[2], down=True, act_fun=act_type, pad=pad, group=4,
                                   bn_mode=bn_mode)  # h/8,  w/8
        self.Conv4 = conv_block_sp(ch_in=enc_ch[2], ch_out=enc_ch[3], down=True, act_fun=act_type, pad=pad, group=4,
                                   bn_mode=bn_mode)  # h/16, w/16
        self.Conv5 = conv_block_sp(ch_in=enc_ch[3], ch_out=enc_ch[4], down=True, act_fun=act_type, pad=pad, group=4,
                                   bn_mode=bn_mode)  # h/32, w/32

        self.conv = nn.Sequential(
            conv(enc_ch[4], dec_ch[4], kernel_size=3, stride=1, bias=True, pad=pad),
            bn(dec_ch[4], bn_mode),
            act(act_type))

        self.Up_conv4 = conv_block_concat(ch_in=dec_ch[4] + enc_ch[3], ch_out=dec_ch[3], act_fun=act_type, pad=pad,
                                          group=4, bn_mode=bn_mode)
        self.Up_conv3 = conv_block_concat(ch_in=dec_ch[3] + enc_ch[2], ch_out=dec_ch[2], act_fun=act_type, pad=pad,
                                          group=4, bn_mode=bn_mode)
        self.Up_conv2 = conv_block_concat(ch_in=dec_ch[2] + enc_ch[1], ch_out=dec_ch[1], act_fun=act_type, pad=pad,
                                          group=4, bn_mode=bn_mode)
        self.Up_conv1 = conv_block_concat(ch_in=dec_ch[1] + enc_ch[0], ch_out=dec_ch[0], act_fun=act_type, pad=pad,
                                          group=4, bn_mode=bn_mode)
        self.Conv_last1 = conv_block_last(dec_ch[0] + img_ch * 4, output_ch * 4, act_fun=act_type, pad=pad, group= 4,
                                          bn_mode=bn_mode)  # concat
        self.Sig = nn.Sigmoid()

    def group_concat(self, d, x, stride1=6, stride2=6):
        return self.concat([d[:,:stride1,:,:], x[:,:stride2,:,:], d[:,stride1:2*stride1,:,:], x[:,stride2:2*stride2,:,:],
                          d[:,2*stride1:3*stride1,:,:], x[:,2*stride2:3*stride2,:,:], d[:,3*stride1:4*stride1,:,:], x[:,3*stride2:4*stride2,:,:]])

    def group_mean(self, out, stride=3):
        return (out[:,:stride] + out[:,stride:2*stride] + out[:,2*stride:3*stride] + out[:,3*stride:4*stride])/4.

    def forward(self, x):
        x = (x - x.mean()) / (x.std())
        x_tmp = self.concat([x, x, x, x])
        # encoding path
        x1 = self.Conv1(x_tmp)  # h/2 w/2
        x2 = self.Conv2(x1)  # h/4 w/4
        x3 = self.Conv3(x2)  # h/8 w/8
        x4 = self.Conv4(x3)  # h/16 w/16
        x5 = self.Conv5(x4)  # h/32 w/32

        x6 = self.conv(x5)

        d5 = self.upsample(x6)  # h/16 w/16
        d5 = self.group_concat(d5, x4)
        d4 = self.Up_conv4(d5)

        d4 = self.upsample(d4)  # h/8  w/8
        d4 = self.group_concat(d4, x3)
        d3 = self.Up_conv3(d4)

        d3 = self.upsample(d3)  # h/4  w/4
        d3 = self.group_concat(d3, x2)
        d2 = self.Up_conv2(d3)

        d2 = self.upsample(d2)  # h/2  w/2
        d2 = self.group_concat(d2, x1)
        d1 = self.Up_conv1(d2)

        d0 = self.upsample(d1)  # h    w

        d0 = self.group_concat(d0, x_tmp, stride2 = self.img_ch)
        d0 = self.Conv_last1(d0)
        out = self.Sig(d0)
        return self.group_mean(out)



class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        inputs_shapes2 = [x.shape[2] for x in [g,x]]
        inputs_shapes3 = [x.shape[3] for x in [g,x]]
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            pass
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            diff2 = (g.size(2) - target_shape2) // 2
            diff3 = (g.size(3) - target_shape3) // 2
            g = g[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3]
            diff2 = (g.size(2) - target_shape2) // 2
            diff3 = (g.size(3) - target_shape3) // 2
            x = x[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3]

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class S2SATnet1(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type = "LeakyReLU", bn_mode = "bn", pad='reflection'):
        super(S2SATnet1, self).__init__()
        enc_ch = [48, 48, 48, 48, 48] # fixed
        dec_ch = [96, 96, 96, 96, 96] # fixed
        print("[*] input/output channel : %d / %d" % (img_ch, output_ch))
        print("[*] act_type : %s" % act_type)
        print("[*] bn_type : %s" % bn_mode)
        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        self.Conv1 = conv_block_sp(ch_in=img_ch, ch_out=enc_ch[0],    down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/2,  w/2
        self.Conv2 = conv_block_sp(ch_in=enc_ch[0], ch_out=enc_ch[1], down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/4,  w/4
        self.Conv3 = conv_block_sp(ch_in=enc_ch[1], ch_out=enc_ch[2], down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/8,  w/8
        self.Conv4 = conv_block_sp(ch_in=enc_ch[2], ch_out=enc_ch[3], down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/16, w/16
        self.Conv5 = conv_block_sp(ch_in=enc_ch[3], ch_out=enc_ch[4], down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/32, w/32

        self.conv = nn.Sequential(
            conv(enc_ch[4], dec_ch[4],  kernel_size=3, stride=1, bias=True, pad = pad),
            bn(dec_ch[4], bn_mode),
            act(act_type))

        self.Up_conv4 = conv_block_concat(ch_in=dec_ch[4] + enc_ch[3], ch_out=dec_ch[3], act_fun= act_type, bn_mode=bn_mode, pad= pad)
        self.Up_conv3 = conv_block_concat(ch_in=dec_ch[3] + enc_ch[2], ch_out=dec_ch[2], act_fun= act_type, bn_mode=bn_mode, pad= pad)
        self.Up_conv2 = conv_block_concat(ch_in=dec_ch[2] + enc_ch[1], ch_out=dec_ch[1], act_fun= act_type, bn_mode=bn_mode, pad= pad)
        self.Up_conv1 = conv_block_concat(ch_in=dec_ch[1] + enc_ch[0], ch_out=dec_ch[0], act_fun= act_type, bn_mode=bn_mode, pad= pad)
        self.Conv_last1 = conv_block_last(dec_ch[0] + img_ch, output_ch, act_fun= act_type, bn_mode=bn_mode, pad = pad) # concat
        self.Sig = nn.Sigmoid()

        self.Att1 = Attention_block(dec_ch[0], enc_ch[0], 48)
        self.Att2 = Attention_block(dec_ch[1], enc_ch[1], 48)
        self.Att3 = Attention_block(dec_ch[2], enc_ch[2], 48)
        self.Att4 = Attention_block(dec_ch[3], enc_ch[3], 48)
        self.Att5 = Attention_block(dec_ch[4], enc_ch[4], 48)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # h/2 w/2
        x2 = self.Conv2(x1) # h/4 w/4
        x3 = self.Conv3(x2) # h/8 w/8
        x4 = self.Conv4(x3) # h/16 w/16
        x5 = self.Conv5(x4) # h/32 w/32

        x6 = self.conv(x5)

        d5 = self.upsample(x6) # h/16 w/16
        x4 = self.Att5(g=d5, x=x4)
        d5 = self.concat([d5, x4])
        d4 = self.Up_conv4(d5)

        d4 = self.upsample(d4) # h/8  w/8
        x3 = self.Att4(g=d4, x=x3)
        d4 = self.concat([d4, x3])
        d3 = self.Up_conv3(d4)

        d3 = self.upsample(d3) # h/4  w/4
        x2 = self.Att3(g=d3, x=x2)
        d3 = self.concat([d3, x2])
        d2 = self.Up_conv2(d3)

        d2 = self.upsample(d2) # h/2  w/2
        x1 = self.Att2(g=d2, x=x1)
        d2 = self.concat([d2, x1])
        d1 = self.Up_conv1(d2)

        d0 = self.upsample(d1) # h    w
        d0 = self.concat([d0, x])
        d0 = self.Conv_last1(d0)
        return self.Sig(d0)

class S2SATnet2(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type = "LeakyReLU", bn_mode = "bn", pad='reflection'):
        super(S2SATnet2, self).__init__()
        enc_ch = [48, 48, 48, 48, 48] # fixed
        dec_ch = [96, 96, 96, 96, 96] # fixed
        print("[*] input/output channel : %d / %d" % (img_ch, output_ch))
        print("[*] act_type : %s" % act_type)
        print("[*] bn_type : %s" % bn_mode)
        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        self.Conv1 = conv_block_sp(ch_in=img_ch, ch_out=enc_ch[0],    down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/2,  w/2
        self.Conv2 = conv_block_sp(ch_in=enc_ch[0], ch_out=enc_ch[1], down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/4,  w/4
        self.Conv3 = conv_block_sp(ch_in=enc_ch[1], ch_out=enc_ch[2], down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/8,  w/8
        self.Conv4 = conv_block_sp(ch_in=enc_ch[2], ch_out=enc_ch[3], down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/16, w/16
        self.Conv5 = conv_block_sp(ch_in=enc_ch[3], ch_out=enc_ch[4], down=True, act_fun= act_type, bn_mode=bn_mode, pad = pad) # h/32, w/32

        self.conv = nn.Sequential(
            conv(enc_ch[4], dec_ch[4],  kernel_size=3, stride=1, bias=True, pad = pad),
            bn(dec_ch[4], bn_mode),
            act(act_type))

        self.Up_conv4 = conv_block_concat(ch_in=dec_ch[4] + enc_ch[3], ch_out=dec_ch[3], act_fun= act_type, bn_mode=bn_mode, pad= pad)
        self.Up_conv3 = conv_block_concat(ch_in=dec_ch[3] + enc_ch[2], ch_out=dec_ch[2], act_fun= act_type, bn_mode=bn_mode, pad= pad)
        self.Up_conv2 = conv_block_concat(ch_in=dec_ch[2] + enc_ch[1], ch_out=dec_ch[1], act_fun= act_type, bn_mode=bn_mode, pad= pad)
        self.Up_conv1 = conv_block_concat(ch_in=dec_ch[1] + enc_ch[0], ch_out=dec_ch[0], act_fun= act_type, bn_mode=bn_mode, pad= pad)
        self.Conv_last1 = conv_block_last(dec_ch[0] + img_ch, output_ch, act_fun= act_type, bn_mode=bn_mode, pad = pad) # concat
        self.Sig = nn.Sigmoid()

        self.Att1 = Attention_block(enc_ch[0], dec_ch[0], 48)
        self.Att2 = Attention_block(enc_ch[1], dec_ch[1], 48)
        self.Att3 = Attention_block(enc_ch[2], dec_ch[2], 48)
        self.Att4 = Attention_block(enc_ch[3], dec_ch[3], 48)
        self.Att5 = Attention_block(enc_ch[4], dec_ch[4], 48)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # h/2 w/2
        x2 = self.Conv2(x1) # h/4 w/4
        x3 = self.Conv3(x2) # h/8 w/8
        x4 = self.Conv4(x3) # h/16 w/16
        x5 = self.Conv5(x4) # h/32 w/32

        x6 = self.conv(x5)

        d5 = self.upsample(x6) # h/16 w/16
        d5 = self.Att5(g=x4, x=d5)
        d5 = self.concat([d5, x4])
        d4 = self.Up_conv4(d5)

        d4 = self.upsample(d4) # h/8  w/8
        d4 = self.Att4(g=x3, x=d4)
        d4 = self.concat([d4, x3])
        d3 = self.Up_conv3(d4)

        d3 = self.upsample(d3) # h/4  w/4
        d3 = self.Att3(g=x2, x=d3)
        d3 = self.concat([d3, x2])
        d2 = self.Up_conv2(d3)

        d2 = self.upsample(d2) # h/2  w/2
        d2 = self.Att2(g=x1, x=d2)
        d2 = self.concat([d2, x1])
        d1 = self.Up_conv1(d2)

        d0 = self.upsample(d1) # h    w
        d0 = self.concat([d0, x])
        d0 = self.Conv_last1(d0)
        return self.Sig(d0)





class Testnet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection'):
        super(Testnet, self).__init__()
        enc_ch = [48, 48, 48, 48, 48]  # fixed
        dec_ch = [48, 48, 48, 48, 48]  # fixed

        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        self.Conv1 = conv_block_sp(ch_in=img_ch, ch_out=enc_ch[0], down=True, act_fun=act_type, pad=pad)  # h/2,  w/2
        self.Conv2 = conv_block_sp(ch_in=enc_ch[0], ch_out=enc_ch[1], down=True, act_fun=act_type, pad=pad)  # h/4,  w/4
        self.Conv3 = conv_block_sp(ch_in=enc_ch[1], ch_out=enc_ch[2], down=True, act_fun=act_type, pad=pad)  # h/8,  w/8
        self.Conv4 = conv_block_sp(ch_in=enc_ch[2], ch_out=enc_ch[3], down=True, act_fun=act_type,
                                   pad=pad)  # h/16, w/16
        self.Conv5 = conv_block_sp(ch_in=enc_ch[3], ch_out=enc_ch[4], down=True, act_fun=act_type,
                                   pad=pad)  # h/32, w/32

        self.conv = nn.Sequential(
            conv(enc_ch[4], dec_ch[4], kernel_size=3, stride=1, bias=True, pad=pad),
            bn(dec_ch[4]),
            act(act_type))

        self.Up_conv4 = conv_block_concat(ch_in=dec_ch[4] + enc_ch[3] + img_ch, ch_out=dec_ch[3], act_fun=act_type, pad=pad)
        self.Up_conv3 = conv_block_concat(ch_in=dec_ch[3] + enc_ch[2] + img_ch, ch_out=dec_ch[2], act_fun=act_type, pad=pad)
        self.Up_conv2 = conv_block_concat(ch_in=dec_ch[2] + enc_ch[1] + img_ch, ch_out=dec_ch[1], act_fun=act_type, pad=pad)
        self.Up_conv1 = conv_block_concat(ch_in=dec_ch[1] + enc_ch[0] + img_ch, ch_out=dec_ch[0], act_fun=act_type, pad=pad)
        self.Conv_last1 = conv_block_last(dec_ch[0] + img_ch, output_ch, act_fun=act_type, pad=pad)  # concat
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x_1 = self.Avgpool(x)
        x_2 = self.Avgpool(x_1)
        x_3 = self.Avgpool(x_2)
        x_4 = self.Avgpool(x_3)

        x1 = self.Conv1(x)  # h/2 w/2
        x2 = self.Conv2(x1) + self.Avgpool(x1)  # h/4 w/4
        x3 = self.Conv3(x2) + self.Avgpool(x2)  # h/8 w/8
        x4 = self.Conv4(x3) + self.Avgpool(x3)  # h/16 w/16
        x5 = self.Conv5(x4) + self.Avgpool(x4)  # h/32 w/32

        x6 = self.conv(x5)

        d5 = self.upsample(x6)  # h/16 w/16
        d5 = self.concat([d5, x4, x_4])
        d4 = self.Up_conv4(d5)

        d4 = self.upsample(d4)  # h/8  w/8
        d4 = self.concat([d4, x3, x_3])
        d3 = self.Up_conv3(d4)

        d3 = self.upsample(d3)  # h/4  w/4
        d3 = self.concat([d3, x2, x_2])
        d2 = self.Up_conv2(d3)

        d2 = self.upsample(d2)  # h/2  w/2
        d2 = self.concat([d2, x1, x_1])
        d1 = self.Up_conv1(d2)

        d0 = self.upsample(d1)  # h    w
        d0 = self.concat([d0, x])
        d0 = self.Conv_last1(d0)
        return self.Sig(d0)

if __name__ == '__main__':
    x = torch.rand([1, 3, 481, 321])
    net = s2s_()

    print(net(x).shape)
