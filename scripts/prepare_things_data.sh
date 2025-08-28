#!/usr/bin/env bash
#SBATCH -A berzelius-2025-35
#SBATCH --mem 100GB
#SBATCH --partition=berzelius-cpu
#SBATCH -t 10:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%J_slurm.err

cd /proj/rep-learning-robotics/users/x_nonra/NeuroLM/
module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate alignvis

data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/
dump_folder=/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2/

for sub in {6..10}; do
    echo "Processing subject $sub"
    python dataset_maker/prepare_THINGS_EEG2.py \
        --data_path $data_path \
        --dump_folder $dump_folder \
        --sub $sub
done
