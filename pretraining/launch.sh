#!/bin/bash

# This is an example script for running the BERT model pretraining on a single node with 8 GPUs.
# You'll most likely have to adjust the script to match your setup.

# This script is called from the slurm.sh script, which sets up the environment and calls this script on each GPU

# Launch script used by slurm scripts, don't invoke directly.

module purge
module load "virtualenv/20.26.2-GCCcore-13.3.0"
source /mimer/NOBACKUP/groups/naiss2024-6-297/envs/babylm/bin/activate

export NCCL_SOCKET_IFNAME=ib0,ib1
export OMP_NUM_THREADS=1

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
echo "Launching on $SLURMD_NODENAME ($SLURM_PROCID/$SLURM_JOB_NUM_NODES)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE," \
     "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"


python -u "$@"
