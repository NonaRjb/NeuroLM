#!/usr/bin/env bash
#SBATCH -A berzelius-2025-35
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%J_slurm.err

CONTAINER="/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif"
data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2"
save_path="/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2/captions"

cd /proj/rep-learning-robotics/users/x_nonra/NeuroLM
apptainer exec --nv $CONTAINER python dataset_maker/generate_captions.py --data_path "${data_path}" --save_path "${save_path}"