#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_vll
#SBATCH -w vll5
#SBATCH -o slurm.out


python gen_back_flow.py --root /data2/local_datasets/vimeo_sequences/ \
    --ckpt models/raft-sintel.pth \
    --gpus 0 1 2 3 4 5 6 7 \
    --batch-size 32 \
    --overwrite 