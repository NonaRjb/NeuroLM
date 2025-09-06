#!/usr/bin/env bash
#SBATCH -A berzelius-2025-278
#SBATCH --mem 200GB
#SBATCH --gpus=2
#SBATCH -t 10:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/NeuroLM/logs/%J_slurm.err


echo "JOB:  ${SLURM_JOB_ID}"
echo "HOST: $(hostname)"
echo ""
export DATA_TAR_DIR=/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2_tar
mkdir -p /scratch/local/x_nonra/data/things_eeg_2
time find $DATA_TAR_DIR -name "*.tar" | xargs -n 1 -P 8 tar -x -C /scratch/local/x_nonra/data/things_eeg_2 -f

ls /scratch/local/x_nonra/data/things_eeg_2/
ls /scratch/local/x_nonra/data/things_eeg_2/processed

export SSL_CERT_FILE=/proj/rep-learning-robotics/users/x_nonra/wandb_cert/cacert.pem
export HF_HOME=/proj/rep-learning-robotics/users/x_nonra/.cache/   
export HF_HUB_CACHE=$HF_HOME/hub/
export HF_DATASETS_CACHE=$HF_HOME/dataset

cd /proj/rep-learning-robotics/users/x_nonra/NeuroLM/

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/neurolm.sif
dataset_dir=/scratch/local/x_nonra/data/things_eeg_2/processed/
out_dir=/proj/rep-learning-robotics/users/x_nonra/NeuroLM/output/
wandb_api_key=$(</proj/rep-learning-robotics/users/x_nonra/NeuroLM/output/wandb/.wandb_key.txt)
wandb_runname=train_vq_J${SLURM_JOB_ID}_$(date +%Y-%m-%d)

apptainer exec --nv \
  --env OMP_NUM_THREADS=2 \
  --env WANDB_API_KEY="$wandb_api_key" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env SSL_CERT_FILE="$SSL_CERT_FILE" \
  "$CONTAINER" \
  torchrun --nnodes=1 --nproc_per_node=1 train_vq.py \
    --dataset_dir "$dataset_dir" \
    --out_dir "$out_dir" \
    --batch_size 32 \
    --warmup_epochs 0 \
    --epochs 60 \
    --patch_size 50 \
    --overlap_size 25 \
    --wandb_log \
    --wandb_project EEG_4M \
    --wandb_api_key $wandb_api_key \
    --wandb_runname "$wandb_runname"


rm -rf /scratch/local/x_nonra/data/things_eeg_2
echo "tmp data removed"
# apptainer exec --nv $CONTAINER OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 train_vq.py \
#     --dataset_dir $dataset_dir \
#     --out_dir $out_dir \
#     --batch_size 32 \
#     --warmup_epochs 0 \
#     --epochs 55 \
#     --wandb_log \
#     --wandb_project EEG_4M \
#     --wandb_runname $wandb_runname \
#     --wandb_api_key $wandb_api_key \