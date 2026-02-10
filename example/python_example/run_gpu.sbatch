#!/bin/bash
#SBATCH --job-name=py-multigpu
#SBATCH --partition=gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

mkdir -p logs


module load gcc/14.2.0
module load cuda/12.8
module load boost/1.86.0   # only if needed

# --- Threading for CPU parts ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPER_NUM_THREADS=$SLRUM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR

echo "Node: $(hostname)"
echo "CPU per task: $SLRUM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# ---- Run your python ----
srun python -u laserPumpCladdingExample.py

