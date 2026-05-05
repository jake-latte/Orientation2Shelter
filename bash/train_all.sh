#!/bin/bash
# Submit cross-animal fold training jobs (one sbatch array task per fold).
#
# Usage:
#   bash bash/train_all.sh [--cv_folds N] [--run_dir PATH] [train.py args...]
#
# Examples:
#   bash bash/train_all.sh --output_mode narrow --embed_dim 128 --epochs 2000
#   bash bash/train_all.sh --output_mode wide --hidden_layers 256,256 --run_dir /path/to/run

set -euo pipefail

N_FOLDS=5
ROOT="/ceph/branco/Jake/training_data"
RUN_DIR=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cv_folds)   N_FOLDS="$2";  PASS_ARGS+=("--cv_folds" "$2"); shift 2 ;;
        --run_dir)    RUN_DIR="$2";  shift 2 ;;
        *)            PASS_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="${ROOT}/runs/all_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${RUN_DIR}/logs"

# Write a transient job script so that $SLURM_ARRAY_TASK_ID is evaluated
# at run time (not at submission time).
JOBSCRIPT=$(mktemp /tmp/train_all_XXXXX.sh)

cat > "$JOBSCRIPT" << 'STATIC'
#!/bin/bash
cd /ceph/scratch/jlaherty/Orientation2Shelter
source /etc/profile.d/modules.sh
module load miniconda
STATIC

# Unquoted heredoc: expands N_FOLDS, RUN_DIR, PASS_ARGS now;
# \$SLURM_ARRAY_TASK_ID is deferred until job execution.
cat >> "$JOBSCRIPT" << DYNAMIC
conda run -p /nfs/ghome/live/jlaherty/anaconda3/envs/O2S \\
    python data-scripts/train.py \\
    --scope all \\
    --cv_folds ${N_FOLDS} \\
    --fold_idx \$SLURM_ARRAY_TASK_ID \\
    --run_dir ${RUN_DIR} \\
    ${PASS_ARGS[*]}
DYNAMIC

chmod +x "$JOBSCRIPT"

sbatch \
    --job-name="train-all" \
    --partition=gpu \
    --nodes=1 \
    --ntasks=4 \
    --mem=32G \
    --gres=gpu:1 \
    --time=1-00:00 \
    --mail-type=END,FAIL \
    --mail-user=h1d2y6c0e0u4q1z8@gatsbyunit.slack.com \
    --output="${RUN_DIR}/logs/fold%a.out" \
    --array="0-$((N_FOLDS - 1))" \
    "$JOBSCRIPT"

rm "$JOBSCRIPT"
echo "Submitted ${N_FOLDS} fold jobs (cross-animal)."
echo "Run dir: ${RUN_DIR}"
