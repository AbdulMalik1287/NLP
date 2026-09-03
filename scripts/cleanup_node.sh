#!/bin/bash
# Remove every trace of this project from the shared node.
#
# Everything the project creates lives under $ROOT (venv, HF cache, data,
# checkpoints, logs), so this is a single delete. Nothing is written to
# ~/.cache, /tmp or system paths.
#
#   bash scripts/cleanup_node.sh --dry-run   # show what would go, and its size
#   bash scripts/cleanup_node.sh --results   # keep results/ and docs/, drop the heavy parts
#   bash scripts/cleanup_node.sh --all       # remove everything

set -uo pipefail
ROOT="${NLP_ROOT:-$HOME/nlp}"
MODE="${1:---dry-run}"

size() { du -sh "$1" 2>/dev/null | cut -f1; }

echo "project root: $ROOT"
[ -d "$ROOT" ] || { echo "nothing there; already clean"; exit 0; }

echo
echo "current usage:"
for d in .venv .cache data models results logs; do
  [ -e "$ROOT/$d" ] && printf "  %-10s %s\n" "$d" "$(size "$ROOT/$d")"
done
printf "  %-10s %s\n" "TOTAL" "$(size "$ROOT")"
echo
echo "disk before:"; df -h "$ROOT" | tail -1

case "$MODE" in
  --dry-run)
    echo
    echo "dry run only. Re-run with --results (keep results) or --all (remove everything)."
    ;;
  --results)
    echo
    echo "removing heavy artifacts, keeping results/ and docs/ ..."
    rm -rf "$ROOT/.venv" "$ROOT/.cache" "$ROOT/data" "$ROOT/models"
    echo "kept: $ROOT/results $ROOT/docs"
    echo "disk after:"; df -h "$ROOT" | tail -1
    ;;
  --all)
    echo
    echo "removing $ROOT entirely ..."
    rm -rf "$ROOT"
    echo "removed."
    echo "disk after:"; df -h "$HOME" | tail -1
    ;;
  *)
    echo "unknown mode: $MODE (use --dry-run, --results or --all)"; exit 1;;
esac

# Slurm jobs outlive a deleted directory; warn if any are still queued.
if command -v squeue >/dev/null 2>&1; then
  n=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "WARNING: $n of your Slurm jobs are still queued/running: squeue -u $USER"
fi
exit 0
