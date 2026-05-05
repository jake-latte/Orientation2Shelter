#!/bin/bash
# Submit per-animal fold training jobs.
# Iterates over all animals (or a single animal via --animal) and submits
# one sbatch array job per animal (one task per fold).
#
# Usage:
#   bash bash/train_animal.sh [--animal A] [--cv_folds N] [--run_dir_root PATH] [train.py args...]
#
# Examples:
#   bash bash/train_animal.sh --output_mode narrow --embed_dim 64
#   bash bash/train_animal.sh --animal 0 --output_mode wide --epochs 2000

set -euo pipefail

N_FOLDS=5
ROOT="/ceph/branco/Jake/training_data"
RUN_DIR_ROOT="${ROOT}/runs"
FILTER_ANIMAL=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cv_folds)      N_FOLDS="$2";       shift 2 ;;  # passed explicitly in job cmd
        --animal)        FILTER_ANIMAL="$2"; shift 2 ;;  # passed via loop var, not PASS_ARGS
        --run_dir_root)  RUN_DIR_ROOT="$2";  shift 2 ;;
        *)               PASS_ARGS+=("$1");  shift ;;
    esac
done

DATESTAMP=$(date +%Y%m%d_%H%M%S)

for animal_dir in "${ROOT}"/*/; do
    [[ -d "$animal_dir" ]] || continue
    animal=$(basename "$animal_dir")

    # Skip if a specific animal was requested
    [[ -n "$FILTER_ANIMAL" && "$animal" != "$FILTER_ANIMAL" ]] && continue

    RUN_DIR="${RUN_DIR_ROOT}/animal_${animal}_${DATESTAMP}"
    mkdir -p "${RUN_DIR}/logs"

    JOBSCRIPT=$(mktemp /tmp/train_animal_XXXXX.sh)

    cat > "$JOBSCRIPT" << 'STATIC'
#!/bin/bash
cd /ceph/scratch/jlaherty/Orientation2Shelter
source /etc/profile.d/modules.sh
module load miniconda
STATIC

    cat >> "$JOBSCRIPT" << DYNAMIC
conda run -p /nfs/ghome/live/jlaherty/anaconda3/envs/O2S \\
    python data-scripts/train.py \\
    --scope animal \\
    --animal ${animal} \\
    --cv_folds ${N_FOLDS} \\
    --fold_idx \$SLURM_ARRAY_TASK_ID \\
    --run_dir ${RUN_DIR} \\
    ${PASS_ARGS[*]}
DYNAMIC

    chmod +x "$JOBSCRIPT"

    sbatch \
        --job-name="train-animal-${animal}" \
        --partition=gpu \
        --nodes=1 \
        --ntasks=4 \
        --mem=32G \
        --gres=gpu:1 \
        --time=0-12:00 \
        --mail-type=END,FAIL \
        --mail-user=h1d2y6c0e0u4q1z8@gatsbyunit.slack.com \
        --output="${RUN_DIR}/logs/fold%a.out" \
        --array="0-$((N_FOLDS - 1))" \
        "$JOBSCRIPT"

    rm "$JOBSCRIPT"
    echo "Submitted ${N_FOLDS} fold jobs for animal ${animal}. Run dir: ${RUN_DIR}"
done
