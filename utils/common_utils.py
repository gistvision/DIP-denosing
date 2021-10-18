import os
import cv2
from utils.blur_utils import *  # blur functions
import torch
import numpy as np

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def cv2_to_torch(cv2_img, dtype=None):
    if dtype == None:
        out = np_to_torch(cv2_img).float() / 255.0
    else:
        out = np_to_torch(cv2_img).type(dtype) / 255.0
    return out

def torch_to_cv2(torch_tensor, clip=True):
    out = torch_to_np(torch_tensor)
    if clip:
        out = np.clip(out, 0, 1)
    return np.squeeze(out * 255).astype(np.uint8)

def save_CHW_np(fname, CHW):
    if len(CHW.shape) == 2:
        cv2.imwrite(fname, CHW)
    else:
        cv2.imwrite(fname, CHW.transpose([1, 2, 0]))

def load_image_pair(fname, task, args):
    """
    1. select degradation to remove.
    We follow the notion (Y = H * X + N)
    :return: X Original image, Y degradation image,
    """
    if task == "deblur":
        clean_img, degradation_img = deblur_loader(fname, args.blur_type, args.sigma)
    elif task == "denoising":
        clean_img, degradation_img = denoise_loader(fname, args.sigma)
    elif task == "poisson":
        clean_img, degradation_img = poisson_loader(fname, args.scale)
    else:
        raise NotImplementedError
    
    print("[!] clean image domain : [%.2f, %.2f]" %(clean_img.min(), clean_img.max()))
    print("[!] noisy image domain : [%.2f, %.2f]" %(degradation_img.min(), degradation_img.max()))
    return clean_img, degradation_img


def poisson_loader(fname, scale):
    img_np = read_image_np(fname)
    img_noisy_np = scale*np.random.poisson(img_np/255.0/scale)* 255.0
    return img_np, img_noisy_np

def denoise_loader(fname, sigma):
    img_np = read_image_np(fname)
    if "mean" in fname:
        img_noisy_np = read_image_np(fname.replace("mean", "real"))
    else:
        img_noisy_np = img_np + np.random.randn(*img_np.shape) * sigma
    return img_np, img_noisy_np

def read_image_np(path):
    """
    :param path: image file name.
    :param gray: Check whether image is gray or color.
    :return:
    """
    img_np = cv2.imread(path, -1)
    if len(img_np.shape) == 2:
        print("[*] read GRAY image.")
        img_np = img_np[np.newaxis,:]
    else:
        print("[*] read COLOR image.")
        img_np = img_np.transpose([2, 0, 1])
    return img_np.astype(np.float) # to added noise.

def read_noise_np(path, sigma):
    # read noise instance same as eSURE.
    try:
        dir_name = os.path.join(os.path.dirname(path), "sigma%s" %  sigma)
        file_name = path.split("/")[-1][:-4]
        file_name = file_name + ".npy"
        new_path = os.path.join(dir_name, file_name)
        noisy_np = np.load(new_path)[0].transpose([2, 0, 1])
    except:
        raise FileNotFoundError
    return noisy_np

def deblur_loader(fname, blur_type, noise_sigma, GRAY_SCALE = False):
    """  Loads an image, and add gaussian blur
    Args:
         fname: path to the image
         blur_type: 'uniform' or 'gauss'
         noise_sigma: noise added after blur
         covert2gray: should we convert to gray scale image?
         plot: will plot the images
    Out:
         dictionary of images and dictionary of psnrs
    """
    BLUR_TYPE = 'gauss_blur' if blur_type == 'g' else 'uniform_blur'
    img_np = read_image_np(fname)        # loadload_and_crop_image    img_pil, 
#     if GRAY_SCALE:
#         img_np = rgb2gray(img_pil)
    blurred = blur(img_np, BLUR_TYPE)  # blur, and the line below adds noise
    blurred = blurred + np.random.normal(scale=noise_sigma, size=blurred.shape)
    return img_np, blurred

if __name__ == "__main__":
    fname = "./testset/CBSD68/3096.png"
    gt, noise = denoise_loader(fname, 10)
    print(gt.shape, gt.min(), gt.max())
    print(noise.shape, noise.min(), noise.max())
    fname = "./testset/BSD68/test001.png"
    gt, noise = denoise_loader(fname, 10)
    print(gt.shape, gt.min(), gt.max())
    print(noise.shape, noise.min(), noise.max())

