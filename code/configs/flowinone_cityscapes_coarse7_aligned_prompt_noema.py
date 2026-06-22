"""Cityscapes COARSE-7 aligned prompt — 7-class scene layout understanding.

Groups 19 classes into 7 semantic groups. Recommends resuming from an
aligned-prompt checkpoint (e.g., the 12000-step aligned continuation).
"""

import importlib.util
from pathlib import Path


def get_config():
    base_path = (Path(__file__).resolve().parents[2] / "e3_cityscapes_teacher"
                 / "configs" / "flowinone_cityscapes_teacher.py")
    spec = importlib.util.spec_from_file_location("teacher_base", base_path)
    base_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(base_module)

    config = base_module.get_config()
    root = Path(__file__).resolve().parents[3]
    exp_root = root / "teacher_understanding_eval" / "e3_cityscapes_teacher"
    data_root = root / "teacher_understanding_eval" / "data" / "cityscapes_coarse7_aligned_visual_prompt_wds"

    config.dataset.train_tar_pattern = f"{data_root}/train/pairs-{{000000..000004}}.tar"
    config.dataset.test_tar_pattern = f"{data_root}/val/pairs-000000.tar"
    config.dataset.vis_image_root = f"{exp_root}/vis_cityscapes_coarse7"
    config.dataset.estimated_samples_per_shard = 600
    config.dataset.output_resample = "nearest"
    config.dataset.resolution = 256

    config.sample.path = f"{exp_root}/samples_cityscapes_coarse7"
    config.sample.sample_steps = 50
    config.sample.cfg = False
    config.sample.scale = 7

    # Resume from aligned prompt continuation (strongest pre-inputmask checkpoint)
    config.train.resume_model_only = True
    config.train.reset_step_on_model_only_resume = 0
    config.train.batch_size = 1
    config.train.log_interval = 100
    config.train.eval_interval = 500
    config.train.save_interval = 250
    config.train.n_samples_eval = 8
    config.train.eval_fid_on_save = False
    config.train.final_eval = False

    config.optimizer.lr = 5.0e-8
    config.lr_scheduler.warmup_steps = 50

    # Use 7-class coarse palette
    config.palette_loss.enabled = False
    config.ema_rate = 0.0
    config.wandb_project = "flowinone_cityscapes_coarse7"
    config.wandb_mode = "disabled"
    config.num_workers = 0

    return config
