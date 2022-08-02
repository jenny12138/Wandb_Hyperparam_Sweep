#!/usr/bin/env bash
#SBATCH --array=1-3
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/mila/j/jiayue.zheng/Projects/BP2T2/Go/sbatch_out/go_bptt/wandb_sweep_go_bptt_%j.out
#SBATCH --error=/home/mila/j/jiayue.zheng/Projects/BP2T2/Go/sbatch_err/go_bptt/wandb_sweep_go_bptt_%j.err
#SBATCH --job-name=wandb_sweep_go_bptt

export LD_DEBUG=files,libs

module load python/3.7 
module load python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0

unset LD_DEBUG

source ~/.venv/rtrl-eprop-venv2/bin/activate

epochs="$1"
gc="$2"
hidden_dim="$3"
lr="$4"
mode="$5"
momentum="$6"
non_linearity="$7"
task="$8"
sweep_id="$9"
sweep_run_name="${10}"
fold=$SLURM_ARRAY_TASK_ID

echo "$sweep_id"
echo "$sweep_run_name"
echo "$fold"

python train_wandb.py --epochs="$epochs" --fold="$fold" --gc="$gc" --hidden_dim="$hidden_dim" --lr="$lr" --mode="$mode" --momentum="$momentum" --non_linearity="$non_linearity" --task="$task" --sweep_id="$sweep_id" --sweep_run_name="$sweep_run_name"
