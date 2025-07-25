#!/usr/bin/env bash
#SBATCH -A berzelius-2025-35
#SBATCH --mem 350GB
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=16
#SBATCH -t 10:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%A_prepare_text.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%A_prepare_text.err

export HF_HOME=/proj/rep-learning-robotics/users/x_nonra/.cache/

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate NeuroLM
python /proj/rep-learning-robotics/users/x_nonra/NeuroLM/text_dataset_maker/prepare.py