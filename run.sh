#!/usr/bin/env bash
set -e

HOME_DIR=''  # Fill in your project root path
MODEL='ViT-B-32'  # MODEL options: ViT-B-32, ViT-L-14
MASK_RATE=0.97
DEVICE='cuda'
USE_QUANT='True'
QUANT_BIT=4
ADDITIONAL_FACTOR=1.0
USE_TRACE_NORM='True'
STEP_SIZE=0.01
BATCH_SIZE=1024

echo "Running ${MODEL}"

python src/ViT.py \
  --home "$HOME_DIR" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --mask_rate "$MASK_RATE" \
  --use_quant "$USE_QUANT" \
  --quant_bit "$QUANT_BIT" \
  --additional_factor "$ADDITIONAL_FACTOR" \
  --batch_size "$BATCH_SIZE" \
  --use_trace_norm "$USE_TRACE_NORM" \
  --step_size "$STEP_SIZE"
