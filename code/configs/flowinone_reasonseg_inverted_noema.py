"""ReasonSeg INVERTED binary mask — white background, black target.

Same inverted approach as RefCOCOg. Reasoning segmentation with
geometry-aligned prompt. Small dataset: 239 train / 200 val.
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
    exp_root = root / "teacher_understanding_eval" / "e7_reasonseg_teacher"
    data_root = root / "teacher_understanding_eval" / "data" / "reasonseg_aligned_inverted_prompt_wds"

    config.dataset.train_tar_pattern = f"{data_root}/train/pairs-000000.tar"
    config.dataset.test_tar_pattern = f"{data_root}/val/pairs-000000.tar"
    config.dataset.vis_image_root = f"{exp_root}/vis_reasonseg_inverted"
    config.dataset.estimated_samples_per_shard = 256
    config.dataset.output_resample = "nearest"
    config.dataset.resolution = 256

    config.sample.path = f"{exp_root}/samples_reasonseg_inverted"
    config.sample.sample_steps = 50
    config.sample.cfg = False
    config.sample.scale = 7

    config.train.batch_size = 1
    config.train.log_interval = 20
    config.train.eval_interval = 100
    config.train.save_interval = 100
    config.train.n_samples_eval = 4
    config.train.eval_fid_on_save = False
    config.train.final_eval = False

    config.optimizer.lr = 5.0e-8
    config.lr_scheduler.warmup_steps = 10

    config.palette_loss.enabled = False
    config.ema_rate = 0.0
    config.wandb_project = "flowinone_reasonseg_inverted"
    config.wandb_mode = "disabled"
    config.num_workers = 0

    return config
