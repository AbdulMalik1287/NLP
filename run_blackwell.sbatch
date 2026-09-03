#!/bin/bash
#SBATCH --job-name=complaint-xlmr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Usage:  sbatch run_blackwell.sbatch scripts/train_transformer.py --model ... --preset node_full
# Adjust --gres/--partition to whatever the cluster actually exposes; a single
# MIG 1g.24gb slice is enough for every experiment except large-batch contrastive.

set -euo pipefail
mkdir -p logs results

module load cuda 2>/dev/null || true
source "${VENV_PATH:-$HOME/envs/complaint}/bin/activate"

export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

nvidia-smi
python scripts/check_env.py || { echo "environment check failed"; exit 1; }

srun python "$@"
