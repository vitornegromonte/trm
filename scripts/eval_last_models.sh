#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
NUM_TEST=${NUM_TEST:-1000}
PASS_AT_K=${PASS_AT_K:-10}
TEMP=${TEMP:-1.0}
TARGET_GLOB="${1:-checkpoints/Sudoku/n[0-9]*}"

results_dir="results"
mkdir -p "$results_dir"

shopt -s nullglob
found=0
for size_dir in $TARGET_GLOB; do
    [ -d "$size_dir" ] || continue
    for run_dir in "$size_dir"/*/; do
        [ -d "$run_dir" ] || continue
        last=$(ls -1 "$run_dir"/step_* 2>/dev/null | grep -v "_all_preds" | sort -V | tail -1)
        [ -n "$last" ] || continue

        label="$(basename "$size_dir")/$(basename "$run_dir")"
        out_json="$results_dir/$(basename "$size_dir")__$(basename "$run_dir").json"
        echo "== $label -> $(basename "$last")"
        "$PYTHON" scripts/evaluate_model.py \
            --checkpoint "$last" \
            --model-type original_trm \
            --domain sudoku \
            --num-test "$NUM_TEST" \
            --pass-at-k "$PASS_AT_K" \
            --temperature "$TEMP" \
            --save-json "$out_json"
        found=1
    done
done

if [ "$found" -eq 0 ]; then
    echo "No checkpoints found under $TARGET_GLOB" >&2
    exit 1
fi

echo
echo "Summary:"
"$PYTHON" - "$results_dir" <<'PY'
import json, pathlib, sys

d = pathlib.Path(sys.argv[1])
for j in sorted(d.glob("*__*.json")):
    r = json.load(open(j))
    name = j.stem.replace("__", "/")
    line = f"  {name}: cell={r['cell_accuracy']:.4f} puzzle={r['puzzle_accuracy']:.4f}"
    for k, v in r.items():
        if k.startswith("pass@"):
            line += f" {k}={v:.4f}"
    print(line)
PY