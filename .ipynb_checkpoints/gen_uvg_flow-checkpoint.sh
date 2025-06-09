#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v5
#SBATCH -o slurm_flow_uvg.out
#SBATCH -e slurm_flow_uvg.err

python3 gen_flow_uvg.py