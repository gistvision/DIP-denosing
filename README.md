# DIP-denosing

This is a code repo for [Rethinking Deep Image Prior for Denoising](https://arxiv.org/abs/2108.12841) (ICCV 2021).

Addressing the relationship between Deep image prior and effective degrees of freedom, DIP-SURE with STE(stochestic temporal ensemble) shows reasonable result on single image denoising.

If you use any of this code, please cite the following publication:

``` Citation
@article{jo2021dipdenoising,
  author  = {Yeonsik Jo, Se young chun,  and Choi, Jonghyun},
  title     = {Rethinking Deep Image Prior for Denoising},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {5087-5096}
}
```

## Working environment

- TITAN Xp
- ubuntu 18.04.4
- pytorch 1.6


**Note:**
Experimental results were not checked in other environments.

## Set-up

- Make your own environment

```bash
conda create --name DIP --file requirements.txt
conda avtivate DIP
pip install tqdm
```

## Inference

- Produce CSet9 result
```bash
bash exp_denoising.sh CSet9 <GPU ID>
```

- For your own data with sigma=25 setup
```bash
mkdir testset/<YOUR_DATASET>
python main.py --dip_type eSURE_new --net_type s2s --exp_tag <EXP_NAME> --optim RAdam --force_steplr --desc sigma25   denoising --sigma 25 --eval_data <YOUR_DATASET>
```

## Browsing experimental result

- We provide reporting code with [invoke](https://www.pyinvoke.org/).
```bash
invoke showtable csv/<exp_type>/<exp_tag> 
```

- Example.
```bash
invoke showtable csv/poisson/MNIST/
PURE_dc_scale001_new                     optimal stopping : 384.30,     31.97/0.02      | ZCSC : 447.60,         31.26/0.02 | STE 31.99/0.02
PURE_dc_scale01_new                      optimal stopping : 94.70,      24.96/0.12      | ZCSC : 144.60,         24.04/0.14 | STE 24.89/0.12
PURE_dc_scale02_new                      optimal stopping : 70.30,      22.92/0.20      | ZCSC : 110.00,         21.82/0.22 | STE 22.83/0.20
<EXEPRIMENTAL NAME>                      optimal stopping :<STEP>,      <PSNR>/<LPIPS>  | ZCSC : <STEP>,      <PSNR>/<LPIPS>| STE <PSNR>/<LPIPS>
```
The reported numbers are PSNR/LPIPS.

## Results in paper
For the result used on paper, please refer [this link](https://drive.google.com/drive/folders/1wAdBUguLTwALFmgmTNNbgwY5zku-c5Pz?usp=sharing).

## SSIM score
For SSIM score of color images, I used matlab code same as the author of [S2S](https://github.com/scut-mingqinchen/self2self).  
This is the demo code I received from the S2S author.  
Thank you Mingqin!
```Matlab
% examples
ref = im2double(imread('gt.png'));
noisy = im2double(imread('noisy.png'));
psnr_result = psnr(ref, noisy);
ssim_result = ssim(ref, noisy);
```

## License

MIT license.

## Contacts

For questions, please send an email to **dustlrdk@gmail.com**