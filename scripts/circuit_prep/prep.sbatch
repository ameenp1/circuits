#!/bin/bash
#SBATCH --job-name=circuit_prep
#SBATCH --output=logs/circuit_prep_%j.out
#SBATCH --error=logs/circuit_prep_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
# Override for parallelism: sbatch --gres=gpu:8 --export=CONFIG=... prep.sbatch
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Generic circuit preparation sbatch template.
# Pass config via --export=CONFIG=configs/texas.yaml
#
# Usage:
#   sbatch --export=CONFIG=configs/texas.yaml scripts/circuit_prep/prep.sbatch

set -e

# Adjust these paths for your cluster setup
cd "$CIRCUITS_ROOT"
source .venv/bin/activate

log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_with_timestamp "Config: $CONFIG"
log_with_timestamp "SLURM Job ID: $SLURM_JOB_ID"
log_with_timestamp "Running on node: $SLURMD_NODENAME"
log_with_timestamp "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if python scripts/circuit_prep/prep.py --config "$CONFIG"; then
    log_with_timestamp "Successfully completed circuit preparation"
else
    exit_code=$?
    log_with_timestamp "ERROR: Circuit prep failed with exit code $exit_code"
    exit $exit_code
fi

log_with_timestamp "Job completed"
