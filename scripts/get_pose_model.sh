#!/usr/bin/env bash
# Fetch a hand-pose ONNX into assets/ for `--model yolo`.
#
# Ultralytics pose exports use the MediaPipe keypoint order, so the strategies and
# the skeleton drawing work unchanged. FP32 is the safe choice for `ort` on the CPU;
# FP16 is half the size but ONNX Runtime has to upcast it per op.
#
# Usage:
#   scripts/get_pose_model.sh              # YOLO26 hand pose (FP32, ~41 MB)
#   scripts/get_pose_model.sh fp16         # same model, ~21 MB
#   scripts/get_pose_model.sh <url>        # any Ultralytics pose export you trained
set -euo pipefail

BASE="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models"
OUT="assets/yolo_hand_pose.onnx"

case "${1:-fp32}" in
  fp32) url="$BASE/yolo26_hand_pose_fp32.onnx" ;;
  fp16) url="$BASE/yolo26_hand_pose_fp16.onnx" ;;
  *)    url="$1" ;;
esac

mkdir -p assets
echo "Downloading $url -> $OUT"
curl -L --fail -o "$OUT" "$url"

echo "Run with: cargo run -- --model yolo --pose-path $OUT"
echo "Cheaper on a CPU: add --pose-input 416 (or 320)."
