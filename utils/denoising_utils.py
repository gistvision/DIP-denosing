import os
from utils.common_utils import *
from skimage.restoration import denoise_nl_means
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np


def non_local_means(noisy_np_img, sigma, fast_mode=True):
    """ get a numpy noisy image
        returns a denoised numpy image using Non-Local-Means
    """ 
    sigma = sigma / 255.
    h = 0.6 * sigma if fast_mode else 0.8 * sigma
    patch_kw = dict(h=h,                   # Cut-off distance, a higher h results in a smoother image
                    sigma=sigma,           # sigma provided
                    fast_mode=fast_mode,   # If True, a fast version is used. If False, the original version is used.
                    patch_size=5,          # 5x5 patches (Size of patches used for denoising.)
                    patch_distance=6,      # 13x13 search area
                    multichannel=False)
    denoised_img = []
    n_channels = noisy_np_img.shape[0]
    for c in range(n_channels):
        denoise_fast = denoise_nl_means(noisy_np_img[c, :, :], **patch_kw)
        denoised_img += [denoise_fast]
    return np.array(denoised_img, dtype=np.float32)

def compare_ssim(a, b):
    if a.shape[0] == 3:
        a = np.mean(a, axis=0)
        b = np.mean(b, axis=0)
    elif a.shape[2] == 3:
        a = np.mean(a, axis=2)
        b = np.mean(b, axis=2)
    else:
        a,b = a[0], b[0]
    return structural_similarity(a,b)



import math
import cv2
# ----------
# PSNR
# ----------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ----------
# SSIM
# ----------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


from .PerceptualSimilarity import PerceptualLoss

def get_lpips(device="cuda"):
    return PerceptualLoss(model='net-lin', net='alex', use_gpu=(device == 'cuda'))

def calculate_lpips(img1_, img2_, LPIPS= None, device="cuda", color= "BGR"):
    if img1_.shape[0] < 3:
        make_color = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        img1 = make_color(img1_[0].astype(np.uint8))
        img2 = make_color(img2_.astype(np.uint8))
    else:
        img1 = img1_.transpose([1,2,0]).astype(np.uint8)
        img2 = img2_.transpose([1,2,0]).astype(np.uint8)
    if LPIPS is None:
        LPIPS = PerceptualLoss(model='net-lin', net='alex', use_gpu=(device == 'cuda'))
    if color == "BGR":
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = torch.tensor(img1.transpose([2,0,1])) / 255.0
    img2 = torch.tensor(img2.transpose([2,0,1])) / 255.0
    if device == "cuda":
        img1 = img1.cuda()
        img2 = img2.cuda()
    return LPIPS(img1, img2, normalize=True).item()


if __name__ == "__main__":
    import numpy as np
    img1 = np.random.rand(255,255)
    img2 = np.random.rand(255,255)
    print(compare_psnr(img1, img2))
    print(compare_ssim(img1, img2))
    min_ = min(img1.min(), img2.min())
    max_ = max(img1.max(), img2.max())
    img1 = ((img1 - min_) / (max_ - min_) * 255).astype(np.uint8)
    img2 = ((img2 - min_) / (max_ - min_) * 255).astype(np.uint8)
    print(compare_psnr(img1, img2), calculate_psnr(img1, img2))
    print(compare_ssim(img1, img2), calculate_ssim(img1, img2))
