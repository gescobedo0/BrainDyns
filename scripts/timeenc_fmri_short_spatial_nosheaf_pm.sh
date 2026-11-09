#!/usr/bin/env bash
# fMRI (PNC) -- SHORT horizon, spatial graph, PARAMETER-MATCHED sheaf ablation.
#
# --no_sheaf removes the restriction-map MLP and map PE (~23k params), so the ODE
# vector field is widened to keep the total equal to the full model:
#   full model (N=400):                  61,705 params
#   no_sheaf + --vf_hidden_dim 165:      61,780 params (+0.12%)
# One shuffle per array task: task i runs shuffle i of TIMEENC_SHUFFLE_SEEDS.
#
#SBATCH --job-name=bd_timeenc_fmri_short_spatial_nosheaf_pm
#SBATCH --partition=gpu,gpu_h200,gpu_rtx6000,gpu_b200,gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err

set -euo pipefail
trap 'echo "[$(date)] ERROR: shuffle_index=${SHUFFLE_INDEX:-<unset>} seed=${SEED_VAL:-<unset>} failed at line ${LINENO} (exit $?)" >&2' ERR

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/BrainDyn-SUMRY}"
cd "${REPO_DIR}"
mkdir -p logs/slurm logs/timeenc_fmri checkpoints/timeenc_fmri
source .venv/bin/activate
source scripts/_timeenc_base.sh

MANIFEST_CSV="${MANIFEST_CSV:-data/manifest_fmri.csv}"
EPOCHS="${EPOCHS:-120}"
LR="${LR:-1e-3}"
SEED="${SEED:-2}"
VF_HIDDEN_DIM="${VF_HIDDEN_DIM:-165}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHUFFLE_INDEX="${SLURM_ARRAY_TASK_ID}"
else
  SHUFFLE_INDEX="${SHUFFLE_INDEX:-0}"
fi
SEED_VAL="${TIMEENC_SHUFFLE_SEEDS[${SHUFFLE_INDEX}]}"

timeenc_graph_flags 2000

SAVE_PATH="checkpoints/timeenc_fmri/timeenc_short_spatial_seed${SEED_VAL}_no_sheaf_pm${RUN_TAG}.pt"
LOG_PATH="logs/timeenc_fmri/timeenc_short_spatial_seed${SEED_VAL}_no_sheaf_pm${RUN_TAG}.log"

CMD=(
  python main.py
  --dataset fmri
  --cohort PNC
  --manifest_csv "${MANIFEST_CSV}"
  --cache
  --x 32 --y 8 --stride 8
  --forecast_mode short
  "${TIMEENC_GRAPH_FLAGS[@]}"
  --epochs "${EPOCHS}"
  --shuffle_seeds "${TIMEENC_SHUFFLE_SEEDS[@]}"
  --shuffle_index "${SHUFFLE_INDEX}"
  --batch_size 64 --num_workers 2
  --hidden_dim 16 --map_hidden_dim 16 --map_mlp_hidden_dim 64 --vf_hidden_dim "${VF_HIDDEN_DIM}"
  "${TIMEENC_FLAGS[@]}"
  --lr "${LR}"
  "${TIMEENC_OPT_FLAGS[@]}"
  --seed "${SEED}"
  --graph_mode "spatial"
  --save_path "${SAVE_PATH}"
  --no_sheaf
)

echo "[$(date)] fMRI timeenc SHORT spatial, param-matched no_sheaf (vf_hidden_dim=${VF_HIDDEN_DIM}): shuffle_index=${SHUFFLE_INDEX} seed=${SEED_VAL} (task ${SLURM_ARRAY_TASK_ID:-n/a})"
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "branch=$(git branch --show-current 2>/dev/null || true)  commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
echo "manifest=${MANIFEST_CSV}"
nvidia-smi -L || true
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
