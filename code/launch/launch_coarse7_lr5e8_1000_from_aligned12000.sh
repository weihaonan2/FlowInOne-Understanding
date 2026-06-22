#!/usr/bin/env bash
# Cityscapes COARSE-7 — 1000 steps, from aligned continuation checkpoint
set -euo pipefail
ROOT="/root/private_data/uvlm"
FLOW_ROOT="${ROOT}/flowinone"
EXP_ROOT="${ROOT}/teacher_understanding_eval/e3_cityscapes_teacher"
CONFIG="${EXP_ROOT}/configs/flowinone_cityscapes_coarse7_aligned_prompt_noema.py"
RESUME="${ROOT}/teacher_understanding_eval/e3_cityscapes_teacher/workdir/aligned_prompt_continue_lr1e7_from_12000/ckpts/12000.ckpt"
WORKDIR="${EXP_ROOT}/workdir/cityscapes_coarse7_lr5e8_1000_from_aligned12000"
mkdir -p "${WORKDIR}"
cd "${FLOW_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_MODE=disabled
exec accelerate launch --num_processes 1 --mixed_precision bf16 run.py \
  --config="${CONFIG}" --workdir="${WORKDIR}" \
  --resume_ckpt_root="${RESUME}" --wandb_mode=disabled \
  --n_steps=1000 --batch_size=1 --log_interval=100 --eval_interval=500 \
  --save_interval=250 --n_samples_eval=8 --mini_batch_size=1 \
  --num_workers=0 --lr=5e-8 --eval_fid_on_save=false --final_eval=false
