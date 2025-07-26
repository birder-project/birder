#!/bin/bash

# ==============================================================================
# DISTRIBUTED TRAINING TEMPLATE FOR BIRDER FRAMEWORK
# ==============================================================================

#SBATCH --job-name=birder_training
#SBATCH --nodes=2                         # Number of nodes
#SBATCH --ntasks-per-node=4               # Number of tasks (GPUs) per node
#SBATCH --gres=gpu:4                      # Number of GPUs per node
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --mem=64G                         # Memory per node
#SBATCH --time=24:00:00                   # Maximum runtime (HH:MM:SS) or (days-HH:MM:SS) e.g. --time=7-00:00:00
#SBATCH --partition=main                  # Partition name (adjust as needed)
#SBATCH --output=logs/slurm_%j.out        # Output log file
#SBATCH --error=logs/slurm_%j.err         # Error log file

# ==============================================================================
# USAGE INSTRUCTIONS
# ==============================================================================
#
# To use this template:
#
# 1. Customize the SBATCH parameters at the top based on your cluster resources
# 2. Modify the training configuration
# 3. Adjust module loading and environment activation commands
# 4. Submit the job with: sbatch distributed_training.sh
#
# Monitor your job with:
# - squeue -u $USER
# - tail -f logs/slurm_<job_id>.out
#
# Cancel the job with:
# - scancel <job_id>
#
# ==============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=== JOB INFORMATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "======================="
echo ""

# System information
echo "--- System Info ---"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "CPU Info: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "CPU Cores: $(nproc)"
echo ""

# Memory information
echo "--- Memory Info ---"
free -h
echo ""

# GPU information
echo "--- GPU Info ---"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo ""

# Load required modules (adjust based on your cluster)
module load python/3.11
module load cuda/12.8

# Set application-specific environment variables
# export LOG_LEVEL=DEBUG

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ib0  # Adjust based on your network interface

# CUDA & performance environment variables
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export CUDA_VISIBLE_DEVICES=0,3

# Activate your virtual environment
echo "=== ACTIVATING ENVIRONMENT ==="
source .venv/bin/activate
echo "Virtual environment activated"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "=============================="
echo ""

# ==============================================================================
# DISTRIBUTED TRAINING COMMAND
# ==============================================================================

echo "=== STARTING DISTRIBUTED TRAINING ==="
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Training start time: $(date)"
echo "====================================="
echo ""

# Launch distributed training
srun python -m birder.scripts.train \
    --network resnet_v1_101 \
    --tag test \
    --opt lamb \
    --lr 0.005 \
    --lr-scheduler cosine \
    --lr-cosine-min 1e-7 \
    --warmup-epochs 5 \
    --epochs 300 \
    --wd 0.02 \
    --grad-accum-steps 4 \
    --mixup-alpha 0.1 \
    --cutmix \
    --aug-type ra \
    --re-prob 0.25 \
    --rgb-mode imagenet \
    --amp --amp-dtype bfloat16 \
    --compile \
    --wds \
    --wds-class-file https://huggingface.co/datasets/birder-project/CUB_200_2011-WDS/resolve/main/classes.txt \
    --wds-info https://huggingface.co/datasets/birder-project/CUB_200_2011-WDS/resolve/main/_info.json

# ==============================================================================
# POST-TRAINING CLEANUP AND REPORTING
# ==============================================================================

TRAINING_EXIT_CODE=$?

echo "=== TRAINING COMPLETED ==="
echo "Training end time: $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
fi

echo "=========================="
