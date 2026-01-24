#!/bin/bash

current_time=$(date +"%H:%M:%S")

TASK_NAME=$1

extract_task_name() {
    local input_string="$1"
    if [[ "$input_string" =~ \.([A-Za-z0-9]+)-task: ]]; then
        TASK_NAME="${BASH_REMATCH[1]}"
    else
        echo "Error: No valid ID found in the checkpoint path."
        exit 1
    fi
}

if [[ "$1" == "-c" ]]; then
    if [[ -n "$2" ]]; then
        extract_task_name "$2"
        echo "$TASK_NAME"
    else
        echo "Error: Missing argument after '-c'."
        exit 1
    fi
else
    TASK_NAME="$1"
fi

MAX_HOURS=48

extract_max_hours() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -max_hours)
                if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                    MAX_HOURS="$2"
                    shift 2
                else
                    echo "Error: -max_hours requires a numeric value."
                    exit 1
                fi
                ;;
            *)
                shift
                ;;
        esac
    done
}

extract_max_hours "$@"

touch build-$TASK_NAME.pbs
echo "#!/bin/bash" >> build-$TASK_NAME.pbs
echo "#PBS -P DATA3888" >> build-$TASK_NAME.pbs
echo "#PBS -l select=1:ncpus=4:ngpus=1:mem=16GB" >> build-$TASK_NAME.pbs
echo "#PBS -l walltime=$MAX_HOURS:00:00" >> build-$TASK_NAME.pbs
echo "#PBS -N $TASK_NAME-$current_time" >> build-$TASK_NAME.pbs
echo "cd /project/DATA3888/gatsby/cueva" >> build-$TASK_NAME.pbs
echo "module load python/3.8.2 magma/2.5.3" >> build-$TASK_NAME.pbs
echo "python3 -m build $@ > build-$TASK_NAME-$current_time.out" >> build-$TASK_NAME.pbs

qsub build-$TASK_NAME.pbs
rm build-$TASK_NAME.pbs