#!/bin/bash
# Local training script for A40 (46 GB) or RTX A6000 (49 GB) GPUs.
# Effective batch size = batch_size_per_gpu × NPROC = 12 × 4 = 48,
# matching the original paper's V100 setup (6 × 8 = 48).
#
# Usage:
#   # All visible GPUs, StyleGAN single MP4:
#   DATA_PATH=/path/to/walk.mp4 bash scripts/pretrain_local.sh
#
#   # Pick specific GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 DATA_PATH=/path/to/walk.mp4 bash scripts/pretrain_local.sh
#
#   # Directory of walks:
#   DATA_PATH=/path/to/walks_dir/ DATA_FORMAT=stylegan bash scripts/pretrain_local.sh
#
#   # Walking-Tours (original single-video mode):
#   DATA_PATH=/path/to/venice.mp4 DATA_FORMAT=single_video bash scripts/pretrain_local.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# GPU selection: derive NPROC from CUDA_VISIBLE_DEVICES when provided,
# otherwise fall back to the explicit NPROC env var (default 4).
# ---------------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '[^[:space:]]')
    export CUDA_VISIBLE_DEVICES
else
    NPROC="${NPROC:-4}"
fi

DATA_PATH="${DATA_PATH:-/path/to/video_or_dir}"
DATA_FORMAT="${DATA_FORMAT:-stylegan}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/dora_run}"
ARCH="${ARCH:-vit_small}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-12}"
# Larger stride for smooth StyleGAN walks; reduce to 30 for fast-moving scenes
STEP_BETWEEN_CLIPS="${STEP_BETWEEN_CLIPS:-120}"

mkdir -p "${OUTPUT_DIR}"

echo "=== DoRA pretraining ==="
echo "  data_path      : ${DATA_PATH}"
echo "  data_format    : ${DATA_FORMAT}"
echo "  arch           : ${ARCH}"
echo "  gpus (NPROC)   : ${NPROC}${CUDA_VISIBLE_DEVICES:+  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})}"
echo "  batch/gpu      : ${BATCH_SIZE}  (total: $((BATCH_SIZE * NPROC)))"
echo "  epochs         : ${EPOCHS}"
echo "  step/clips     : ${STEP_BETWEEN_CLIPS}"
echo "  output_dir     : ${OUTPUT_DIR}"
echo "========================"

torchrun --nproc_per_node="${NPROC}" main.py \
  --arch "${ARCH}" \
  --data_path "${DATA_PATH}" \
  --data_format "${DATA_FORMAT}" \
  --output_dir "${OUTPUT_DIR}" \
  --optimizer adamw \
  --use_bn_in_head False \
  --out_dim 65536 \
  --batch_size_per_gpu "${BATCH_SIZE}" \
  --local_crops_number 6 \
  --epochs "${EPOCHS}" \
  --num_workers 8 \
  --lr 0.0005 \
  --min_lr 0.00001 \
  --norm_last_layer False \
  --warmup_teacher_temp_epochs 30 \
  --weight_decay 0.04 \
  --weight_decay_end 0.4 \
  --frame_per_clip 8 \
  --step_between_clips "${STEP_BETWEEN_CLIPS}"
