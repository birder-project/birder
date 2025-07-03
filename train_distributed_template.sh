#!/bin/bash

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
# DISTRIBUTED TRAINING TEMPLATE FOR BIRDER FRAMEWORK
# ==============================================================================
# This template demonstrates how to run distributed training using Slurm
# with the Birder deep learning framework.
#
# Key Features:
# - Multi-node, multi-GPU support
# - Automatic distributed environment setup
# - Configurable training parameters
# - Error handling and logging
# - Easy customization for different experiments
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
echo "========================"

# Load required modules (adjust based on your cluster)
module load python/3.11
module load cuda/12.8

# Activate your virtual environment
source activate birder

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0 # Adjust based on your network interface

# Optional: Set CUDA visible devices and other optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==============================================================================
# DISTRIBUTED TRAINING COMMAND
# ==============================================================================

echo "=== STARTING DISTRIBUTED TRAINING ==="
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "========================================="

# Launch distributed training
srun python -m birder.scripts.train_dino_v2 \
    --network rope_vit_reg8_so150m_p14_ap \
    --ibot-separate-head \
    --dino-out-dim 131072 \
    --ibot-out-dim 131072 \
    --head-bottleneck-dim 384 \
    --centering sinkhorn_knopp \
    --local-crop-size 98 \
    --opt adamw \
    --lr 0.0002 \
    --lr-scheduler-update iter \
    --lr-scheduler cosine \
    --lr-cosine-min 1e-6 \
    --epochs 400 \
    --warmup-epochs 40 \
    --batch-size 64 \
    --wd 0.04 \
    --wd-end 0.2 \
    --clip-grad-norm 3 \
    --model-config drop_path_rate=0.3 \
    --amp --amp-dtype bfloat16 \
    --compile \
    --rgb-mode none \
    --wds --wds-info /mnt/data/ssl_packed/_info.json

# ==============================================================================
# POST-TRAINING CLEANUP AND REPORTING
# ==============================================================================

TRAINING_EXIT_CODE=$?

echo "=== TRAINING COMPLETED ==="
echo "Exit code: $TRAINING_EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
fi

echo "=========================="

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
# ==============================================================================
