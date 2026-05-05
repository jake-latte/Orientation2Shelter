#!/bin/bash
# Submit per-session fold training jobs.
# Iterates over all (animal, session) pairs (or a specific one) and submits
# one sbatch array job per session (one task per fold).
#
# Usage:
#   bash bash/train_session.sh [--animal A] [--session S] [--cv_folds N] [--run_dir_root PATH] [train.py args...]
#
# Examples:
#   bash bash/train_session.sh --output_mode narrow --hidden_layers 64,64
#   bash bash/train_session.sh --animal 0 --session 1 --epochs 5000
#   bash bash/train_session.sh --animal 2 --output_mode wide

set -euo pipefail

N_FOLDS=5
ROOT="/ceph/branco/Jake/training_data"
RUN_DIR_ROOT="${ROOT}/runs"
FILTER_ANIMAL=""
FILTER_SESSION=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cv_folds)      N_FOLDS="$2";         shift 2 ;;  # passed explicitly in job cmd
        --animal)        FILTER_ANIMAL="$2";   shift 2 ;;  # passed via loop var, not PASS_ARGS
        --session)       FILTER_SESSION="$2";  shift 2 ;;  # passed via loop var, not PASS_ARGS
        --run_dir_root)  RUN_DIR_ROOT="$2";    shift 2 ;;
        *)               PASS_ARGS+=("$1");    shift ;;
    esac
done

DATESTAMP=$(date +%Y%m%d_%H%M%S)

for animal_dir in "${ROOT}"/*/; do
    [[ -d "$animal_dir" ]] || continue
    animal=$(basename "$animal_dir")
    [[ -n "$FILTER_ANIMAL" && "$animal" != "$FILTER_ANIMAL" ]] && continue

    for session_dir in "${animal_dir}"*/; do
        [[ -d "$session_dir" ]] || continue
        [[ -f "${session_dir}/data.csv" ]] || continue
        session=$(basename "$session_dir")
        [[ -n "$FILTER_SESSION" && "$session" != "$FILTER_SESSION" ]] && continue

        RUN_DIR="${RUN_DIR_ROOT}/session_${animal}_${session}_${DATESTAMP}"
        mkdir -p "${RUN_DIR}/logs"

        JOBSCRIPT=$(mktemp /tmp/train_session_XXXXX.sh)

        cat > "$JOBSCRIPT" << 'STATIC'
#!/bin/bash
cd /ceph/scratch/jlaherty/Orientation2Shelter
source /etc/profile.d/modules.sh
module load miniconda
STATIC

        cat >> "$JOBSCRIPT" << DYNAMIC
conda run -p /nfs/ghome/live/jlaherty/anaconda3/envs/O2S \\
    python data-scripts/train.py \\
    --scope session \\
    --animal ${animal} \\
    --session ${session} \\
    --cv_folds ${N_FOLDS} \\
    --fold_idx \$SLURM_ARRAY_TASK_ID \\
    --run_dir ${RUN_DIR} \\
    ${PASS_ARGS[*]}
DYNAMIC

        chmod +x "$JOBSCRIPT"

        sbatch \
            --job-name="train-ses-${animal}-${session}" \
            --partition=gpu \
            --nodes=1 \
            --ntasks=4 \
            --mem=16G \
            --gres=gpu:1 \
            --time=0-06:00 \
            --mail-type=END,FAIL \
            --mail-user=h1d2y6c0e0u4q1z8@gatsbyunit.slack.com \
            --output="${RUN_DIR}/logs/fold%a.out" \
            --array="0-$((N_FOLDS - 1))" \
            "$JOBSCRIPT"

        rm "$JOBSCRIPT"
        echo "Submitted ${N_FOLDS} fold jobs for session ${animal}/${session}. Run dir: ${RUN_DIR}"
    done
done
