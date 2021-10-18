#!/usr/bin/env bash

echo task $1 GPU_ID $2
export CUDA_VISIBLE_DEVICES=$2

# [ ] need space.
if [ $1 == "dip" ]; then
    tag=dip_set9
    python main.py --dip_type dip    --net_type s2s  --exp_tag $tag  --desc sigma15  denoising   --sigma 15
    python main.py --dip_type dip    --net_type s2s  --exp_tag $tag  --desc sigma25  denoising   --sigma 25
    python main.py --dip_type dip    --net_type s2s  --exp_tag $tag   --desc sigma50  denoising   --sigma 50

elif [ $1 == "ablation" ]; then
    tag=ablation
    python main.py --dip_type dip    --net_type s2s  --exp_tag $tag  --desc sigma25  denoising   --sigma 25
    python main.py --dip_type dip_sure    --net_type s2s  --exp_tag $tag  --desc sigma25  denoising   --sigma 25
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag CSet9 --optim RAdam --force_steplr --desc sigma25   denoising --sigma 25

elif [ $1 == "CSet9" ]; then
    tag=CSet9
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag CSet9 --optim RAdam --force_steplr --desc sigma15   denoising --sigma 15
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag CSet9 --optim RAdam --force_steplr --desc sigma25   denoising --sigma 25
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag CSet9 --optim RAdam --force_steplr --desc sigma50   denoising --sigma 50

elif [ $1 == "set12" ]; then
    tag=Set12
    python main.py --dip_type eSURE_uniform    --gray --net_type s2s --exp_tag $tag --optim RAdam --force_steplr  --sigma_z 0.3 --desc sigma15 denoising --sigma 15 --eval_data Set12
    python main.py --dip_type eSURE_uniform    --gray --net_type s2s --exp_tag $tag --optim RAdam --force_steplr  --sigma_z 0.3 --desc sigma25 denoising --sigma 25 --eval_data Set12
    python main.py --dip_type eSURE_uniform    --gray --net_type s2s --exp_tag $tag --optim RAdam --force_steplr  --sigma_z 0.3 --desc sigma50 denoising --sigma 50 --eval_data Set12

elif [ $1 == "McM" ]; then
    tag=McM
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma15 denoising --sigma 15 --eval_data McM
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma25 denoising --sigma 25 --eval_data McM
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma50 denoising --sigma 50 --eval_data McM

elif [ $1 == "kodak" ]; then
    tag=kodak
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma15  denoising --sigma 15 --eval_data Kodak
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma25  denoising --sigma 25 --eval_data Kodak
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma50  denoising --sigma 50 --eval_data Kodak

elif [ $1 == "CBSD" ]; then
    tag=CBSD
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma15   denoising --sigma 15 --eval_data CBSD68
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma25   denoising --sigma 25 --eval_data CBSD68
    python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag $tag --optim RAdam --force_steplr --desc sigma50   denoising --sigma 50 --eval_data CBSD68

elif [ $1 == "BSD" ]; then
    tag=BSD
    python main.py --dip_type eSURE_uniform    --gray --net_type s2s --exp_tag $tag --optim RAdam --force_steplr  --sigma_z 0.3 --desc sigma15   denoising --sigma 15 --eval_data BSD68
    python main.py --dip_type eSURE_uniform    --gray --net_type s2s --exp_tag $tag --optim RAdam --force_steplr  --sigma_z 0.3 --desc sigma25   denoising --sigma 25 --eval_data BSD68
    python main.py --dip_type eSURE_uniform    --gray --net_type s2s --exp_tag $tag --optim RAdam --force_steplr  --sigma_z 0.3 --desc sigma50   denoising --sigma 50 --eval_data BSD68

elif [ $1 == "DIP_MNIST" ]; then
    tag=DIP_MNIST
    python main.py --dip_type dip    --gray --running_avg_ratio 0.9 --exp_tag $tag --optim RAdam --force_steplr --desc scale001  poisson --scale 0.01 --eval_data MNIST
    python main.py --dip_type dip    --gray --running_avg_ratio 0.9 --exp_tag $tag --optim RAdam --force_steplr --desc scale01   poisson --scale 0.1 --eval_data MNIST
    python main.py --dip_type dip    --gray --running_avg_ratio 0.9 --exp_tag $tag --optim RAdam --force_steplr --desc scale02   poisson --scale 0.2 --eval_data MNIST

elif [ $1 == "MNIST" ]; then
    tag=MNIST
    python main.py --dip_type PURE_dc    --gray --running_avg_ratio 0.9 --net_type s2s_normal --exp_tag $tag --optim RAdam --desc scale001  poisson --scale 0.01 --eval_data MNIST
    python main.py --dip_type PURE_dc    --gray --running_avg_ratio 0.9 --net_type s2s_normal --exp_tag $tag --optim RAdam --desc scale01   poisson --scale 0.1 --eval_data MNIST
    python main.py --dip_type PURE_dc    --gray --running_avg_ratio 0.9 --net_type s2s_normal --exp_tag $tag --optim RAdam --desc scale02   poisson --scale 0.2 --eval_data MNIST

else
    echo wrong task name
fi
