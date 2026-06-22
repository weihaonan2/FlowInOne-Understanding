#!/usr/bin/env bash
# ReasonSeg INVERTED binary mask — 500 steps, from original teacher
set -euo pipefail
ROOT="/root/private_data/uvlm"
FLOW_ROOT="${ROOT}/flowinone"
EXP_ROOT="${ROOT}/teacher_understanding_eval/e7_reasonseg_teacher"
CONFIG="${EXP_ROOT}/configs/flowinone_reasonseg_inverted_noema.py"
WORKDIR="${EXP_ROOT}/workdir/reasonseg_inverted_lr5e8_500_from_original"
mkdir -p "${WORKDIR}"
cd "${FLOW_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled
exec accelerate launch --num_processes 1 --mixed_precision bf16 run.py \
  --config="${CONFIG}" --workdir="${WORKDIR}" --wandb_mode=disabled \
  --n_steps=500 --batch_size=1 --log_interval=50 --eval_interval=100 \
  --save_interval=100 --n_samples_eval=4 --mini_batch_size=1 \
  --num_workers=0 --lr=5e-8 --eval_fid_on_save=false --final_eval=false
