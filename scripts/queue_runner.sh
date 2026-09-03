#!/bin/bash
# Pull jobs from a shared queue file until it is empty. One worker per GPU slice,
# so the GPUs never idle between runs.
#
#   bash scripts/queue_runner.sh <queue-file> <cuda-device>
#
# flock makes the pop atomic, so two workers never take the same job.

set -uo pipefail
QUEUE="${1:?usage: queue_runner.sh <queue-file> <cuda-device>}"
DEV="${2:?usage: queue_runner.sh <queue-file> <cuda-device>}"
LOCK="$QUEUE.lock"
DONE="$QUEUE.done"
FAIL="$QUEUE.failed"

cd "$(dirname "$0")/.." || exit 1
. scripts/node_env.sh

while true; do
    JOB=$(flock "$LOCK" bash -c "
        head -n1 '$QUEUE' 2>/dev/null;
        tail -n +2 '$QUEUE' > '$QUEUE.tmp' 2>/dev/null && mv '$QUEUE.tmp' '$QUEUE'
    ")
    [ -z "$JOB" ] && { echo "[dev$DEV] queue empty, exiting"; break; }
    case "$JOB" in \#*) continue;; esac

    NAME=$(echo "$JOB" | grep -oP '(?<=--tag )\S+' || echo job)
    MODEL=$(echo "$JOB" | grep -oP '(?<=--model )\S+' | tr '/' '_' || echo model)
    LOG="logs/${NAME}_${MODEL}_dev${DEV}.log"
    echo "[dev$DEV] $(date +%H:%M:%S) START $NAME $MODEL"
    START=$(date +%s)

    if CUDA_VISIBLE_DEVICES="$DEV" eval "$JOB" > "$LOG" 2>&1; then
        ELAPSED=$(( $(date +%s) - START ))
        RESULT=$(grep -a 'TEST macro-F1' "$LOG" | tail -1)
        flock "$LOCK" bash -c "echo '[dev$DEV] ${ELAPSED}s $NAME $MODEL :: $RESULT' >> '$DONE'"
        echo "[dev$DEV] $(date +%H:%M:%S) DONE  ($((ELAPSED/60))m) $RESULT"
    else
        flock "$LOCK" bash -c "echo '[dev$DEV] FAILED $NAME $MODEL (see $LOG)' >> '$FAIL'"
        echo "[dev$DEV] $(date +%H:%M:%S) FAILED $NAME $MODEL"
    fi
done
