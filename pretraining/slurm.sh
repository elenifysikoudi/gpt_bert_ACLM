#!/bin/bash

#SBATCH -A NAISS2025-5-180
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:2
#SBATCH --ntasks-per-node=2
#SBATCH -p alvis
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=bert-%j.out


# This is an example script for running the BERT model pretraining on a single node with 8 GPUs.
# You'll most likely have to adjust the script to match your setup.

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9996
export WORLD_SIZE=$SLURM_NTASKS
echo $WORLD_SIZE

CONTAINER="/apps/containers/PyTorch/PyTorch-NGC-latest.sif"
SING_BIND="/mimer/NOBACKUP/groups/naiss2024-6-297/gpt-bert/pretraining/"

set -euo pipefail

CMD="train_ACLM.py"

echo $CMD
echo "START $SLURM_JOBID: $(date)"

srun \
    --label \
    singularity exec \
    -B "$SING_BIND:/container/path" \
    "$CONTAINER" \
    bash /container/path/launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"

