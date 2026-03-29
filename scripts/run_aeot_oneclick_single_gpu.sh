#!/usr/bin/env bash
set -euo pipefail

# Single-GPU one-click generator wrapper for current host.
# Usage:
#   bash scripts/run_aeot_oneclick_single_gpu.sh /abs/path/to/ae.ckpt [run_name]

AE_CKPT="${1:-}"
RUN_NAME="${2:-quick_recall}"

if [[ -z "${AE_CKPT}" ]]; then
  echo "Usage: bash scripts/run_aeot_oneclick_single_gpu.sh /abs/path/to/ae.ckpt [run_name]"
  exit 1
fi

if [[ ! -f "${AE_CKPT}" ]]; then
  echo "AE checkpoint not found: ${AE_CKPT}"
  exit 1
fi

ROOT="/home/zky/PyTorch-VAE"
FEATURES_PT="/home/zky/AE-OT/results_curves/features_5w.pt"
OT_H="/home/zky/AE-OT/results_curves/h.pt"
OT_ROOT="/home/zky/AE-OT"
OUT_ROOT="${ROOT}/results/aeot_runs"

if [[ ! -f "${FEATURES_PT}" ]]; then
  echo "features_5w.pt not found: ${FEATURES_PT}"
  exit 1
fi
if [[ ! -f "${OT_H}" ]]; then
  echo "h.pt not found: ${OT_H}"
  exit 1
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES=0 python scripts/run_aeot_end2end.py \
  --ae_config "${ROOT}/configs/stage1_ae.yaml" \
  --ae_ckpt "${AE_CKPT}" \
  --features_pt "${FEATURES_PT}" \
  --ot_h "${OT_H}" \
  --out_root "${OUT_ROOT}" \
  --run_name "${RUN_NAME}" \
  --n_generate 2000 \
  --num_gen_x 50000 \
  --ot_bat_size_n 10000 \
  --ot_thresh 0.3 \
  --decode_batch_size 128 \
  --min_length 2 \
  --min_pairwise_dist 2.0 \
  --neighbor_exclude 2 \
  --ot_root "${OT_ROOT}" \
  --gpu_id 0 \
  --select_random \
  --seed 42
