#!/usr/bin/env bash
#SBATCH -A berzelius-2025-35
#SBATCH --mem 350GB
#SBATCH --gpus=8
#SBATCH -t 10:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%A_%a_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%A_%a_slurm.err


cd /proj/rep-learning-robotics/users/x_nonra/NeuroLM/
module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate NeuroLM

dataset_dir=/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2/processed/
out_dir=/proj/rep-learning-robotics/users/x_nonra/NeuroLM/output/
wandb_api_key=REMOVED_API_KEY

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 train_vq.py \
    --dataset_dir $dataset_dir \
    --out_dir $out_dir \
    --wandb_log \
    --wandb_project EEG_4M \
    --wandb_runname test_vq_train \
    --wandb_api_key $wandb_api_key \