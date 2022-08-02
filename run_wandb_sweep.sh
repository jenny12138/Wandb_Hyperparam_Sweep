#!/usr/bin/env bash
#SBATCH --array=1-80
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mem=2GB
#SBATCH --time=0:10:00
#SBATCH --output=/home/mila/j/jiayue.zheng/Projects/BP2T2/Go/sweep_central_sbatch_out/go_bptt/central_sweep_%j.out
#SBATCH --error=/home/mila/j/jiayue.zheng/Projects/BP2T2/Go/sweep_central_sbatch_err/go_bptt/central_sweep_%j.err
#SBATCH --job-name=central_sweep_go_bptt

sweep_id="$1"

module load python/3.7 
module load python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source ~/.venv/rtrl-eprop-venv2/bin/activate

wandb agent --count 1 jenn12138/BP2T2-Go/"$sweep_id"
