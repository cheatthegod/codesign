#!/usr/bin/env bash
# Wait for step 30000 checkpoint, then restart with optimized config
set -euo pipefail

DISCO_DIR="/home/ubuntu/cqr_files/protein_design/COT_enzyme_design/DISCO"
CKPT="${DISCO_DIR}/train_output_conditional/checkpoints/disco_step_30000.pt"
LOG="${DISCO_DIR}/train_output_conditional/train.log.fast"

echo "[monitor] Waiting for checkpoint: ${CKPT}"
echo "[monitor] Checking every 60 seconds..."

while [ ! -f "${CKPT}" ]; do
    # Show latest step from log
    LATEST=$(tail -1 "${DISCO_DIR}/train_output_conditional/train.log.resume" 2>/dev/null | grep -oP 'Step \d+' || echo "unknown")
    echo "[monitor] $(date '+%H:%M:%S') - ${LATEST} - waiting..."
    sleep 60
done

# Wait a bit for checkpoint write to complete
echo "[monitor] Checkpoint detected! Waiting 30s for write to complete..."
sleep 30

# Verify checkpoint size (should be ~5.9GB)
CKPT_SIZE=$(stat -c%s "${CKPT}" 2>/dev/null || echo 0)
if [ "${CKPT_SIZE}" -lt 1000000000 ]; then
    echo "[monitor] ERROR: Checkpoint too small (${CKPT_SIZE} bytes), aborting"
    exit 1
fi
echo "[monitor] Checkpoint OK: $(du -h "${CKPT}" | cut -f1)"

# Kill current training
echo "[monitor] Stopping current training..."
pkill -f "runner/train.py" || true
sleep 10

# Verify processes stopped
if pgrep -f "runner/train.py" > /dev/null 2>&1; then
    echo "[monitor] Force killing remaining processes..."
    pkill -9 -f "runner/train.py" || true
    sleep 5
fi

echo "[monitor] Starting optimized training..."
cd "${DISCO_DIR}"

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

# Launch optimized training
nohup torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    runner/train.py \
    --config-name=train_conditional \
    > "${LOG}" 2>&1 &

echo "[monitor] Optimized training started! PID: $!"
echo "[monitor] Log: ${LOG}"
echo "[monitor] Done."
