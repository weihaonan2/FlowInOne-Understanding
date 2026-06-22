#!/usr/bin/env bash
# GPU0: RefCOCOg INVERTED binary mask (white bg, black target)
set -euo pipefail
ROOT="/root/private_data/uvlm"
FLOW_ROOT="${ROOT}/flowinone"
EXP_ROOT="${ROOT}/teacher_understanding_eval/e6_refcocog_teacher"
CONFIG="${EXP_ROOT}/configs/flowinone_refcocog_inverted_binary_noema.py"
WORKDIR="${EXP_ROOT}/workdir/refcocog_inverted_lr1e7_1000_from_original"
mkdir -p "${WORKDIR}"
cd "${FLOW_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
exec accelerate launch --num_processes 1 --mixed_precision bf16 run.py \
  --config="${CONFIG}" --workdir="${WORKDIR}" --wandb_mode=disabled \
  --n_steps=1000 --batch_size=1 --log_interval=100 --eval_interval=250 \
  --save_interval=250 --n_samples_eval=8 --mini_batch_size=1 --num_workers=0 \
  --lr=1e-7 --eval_fid_on_save=false --final_eval=false
