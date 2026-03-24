#!/bin/bash
# Comprehensive benchmark: different batch sizes, with/without augmentation
# Uses 1 epoch only to avoid epoch-boundary DataLoader issues

CACHE_DIR="/tmp/pluto_augment_cache_20k"
PYTHON="/opt/conda/envs/pluto/bin/python"
TORCHRUN="/opt/conda/envs/pluto/bin/torchrun"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/benchmark_augmentation.py"
OUTDIR="$SCRIPT_DIR/benchmark_results/augment_comparison"
mkdir -p "$OUTDIR"

WARMUP=10
MAX_STEPS=120
EPOCHS=1
WORKERS=8

echo "============================================================"
echo "  Augmentation Impact Benchmark"
echo "  Cache: $CACHE_DIR"
echo "  Warmup: $WARMUP steps, Max steps/epoch: $MAX_STEPS, Epochs: $EPOCHS"
echo "============================================================"

PORT=29500
for BS in 8 16 32 64; do
    for AUG in "--no-augment" "--augment"; do
        if [ "$AUG" = "--augment" ]; then
            LABEL="aug"
        else
            LABEL="noaug"
        fi
        OUTFILE="$OUTDIR/bs${BS}_${LABEL}.txt"
        echo ""
        echo ">>> Running: bs=$BS $AUG (port=$PORT)"
        $TORCHRUN --nproc_per_node=8 --master-port=$PORT \
            $SCRIPT --mode train \
            --cache-dir $CACHE_DIR \
            --batch-size $BS \
            --num-workers $WORKERS \
            --warmup-steps $WARMUP \
            --max-steps-per-epoch $MAX_STEPS \
            --num-epochs $EPOCHS \
            $AUG \
            2>&1 | tee "$OUTFILE"
        PORT=$((PORT + 1))
        echo ">>> Saved to $OUTFILE"
    done
done

echo ""
echo "============================================================"
echo "  All benchmarks complete. Results in $OUTDIR"
echo "============================================================"
