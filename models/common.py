import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler
from .iterBN import IterNorm

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
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


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)

class Tanh(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)

class Sin(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun[:3] == 'ELU':
            if len(act_fun)> 3:
                param = float(act_fun[3:])
                return nn.ELU(param, inplace=True)
            return nn.ELU(inplace=True)
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'tanh':
            return Tanh()
        elif act_fun == 'sine':
            return Sin()
        elif act_fun == 'soft':
            return nn.Softplus()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features, mode = "bn"):
    if mode == "bn":
        return nn.BatchNorm2d(num_features)
    elif mode == "bn_kai":
        return nn.BatchNorm2d(num_features, momentum=0.9, eps=1e-4)
    elif mode == "In":
        return nn.InstanceNorm2d(num_features, affine= True)
    elif mode == "None":
        return nn.Sequential()

    elif mode == "bn_kai7":
        return nn.BatchNorm2d(num_features, momentum=0.7, eps=1e-4)
    elif mode == "bn_kai5":
        return nn.BatchNorm2d(num_features, momentum=0.5, eps=1e-4)
    elif mode == "bn_eps":
        return nn.BatchNorm2d(num_features, eps=1e-4)
    elif mode == "iterbn":
        return IterNorm(num_features)

    elif mode == "groupNorm":
        return nn.GroupNorm(4, num_features)

    else :
        return None


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', group = 1, downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad,  groups= group,bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)
