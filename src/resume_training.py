"""Resume and continue YOLOv11 training from checkpoints.

Provides utilities for resuming interrupted training, modifying
hyperparameters on resume, and validating resumed models.
"""

import gc
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from ultralytics import YOLO


def resume_tuberculosis_training(
    checkpoint_path: Optional[str] = None,
    additional_epochs: int = 50,
    data_yaml: str = "/kaggle/working/yolov11_dataset_1k_guaranteed/dataset.yaml",
    project_dir: str = "tuberculosis_optimized_v11",
    run_name: str = "tb_best_yolo11m_v1_continued",
    batch_size: int = 8,
) -> Optional[Any]:
    """Resume training from a checkpoint.

    Args:
        checkpoint_path: Path to ``last.pt`` or ``best.pt``.
        additional_epochs: Extra epochs to train.
        data_yaml: Dataset YAML config path.
        project_dir: Output project directory.
        run_name: Sub-directory name for resumed run.
        batch_size: Batch size for resumed training.

    Returns:
        Ultralytics results on success, *None* on failure.
    """
    if checkpoint_path is None:
        # Search default locations
        candidates = [
            f"{project_dir}/tb_best_yolo11m_v1/weights/last.pt",
            f"{project_dir}/tb_best_yolo11m_v1/weights/best.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found")
        return None

    print(f"Resuming from: {checkpoint_path}")
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = YOLO(checkpoint_path)
        results = model.train(
            data=data_yaml,
            epochs=additional_epochs,
            batch=batch_size,
            imgsz=640,
            device=0,
            project=project_dir,
            name=run_name,
            exist_ok=True,
            resume=True,
            optimizer="AdamW",
            lr0=0.0005,
            patience=30,
            cos_lr=True,
            save=True,
            save_period=5,
            plots=True,
            verbose=True,
        )
        print(f"\n✅ Resume training completed — results in {project_dir}/{run_name}")
        return results

    except Exception as exc:
        print(f"\n❌ Resume failed: {exc}")
        torch.cuda.empty_cache()
        gc.collect()
        return None


def smart_batch_optimizer_resume(
    checkpoint_path: str,
    target_memory_usage: float = 0.75,
) -> int:
    """Find optimal batch size when resuming from a checkpoint.

    More conservative than initial training (75% target vs 85%).

    Args:
        checkpoint_path: Path to checkpoint weights.
        target_memory_usage: Target GPU memory fraction.

    Returns:
        Recommended batch size.
    """
    if not torch.cuda.is_available():
        return 4

    total_mem = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
    if total_mem >= 16:
        return 12
    if total_mem >= 8:
        return 6
    return 4


def validate_resumed_model(model_path: str, data_yaml: str) -> None:
    """Quick validation of a resumed/trained model.

    Args:
        model_path: Path to model weights.
        data_yaml: Dataset YAML config path.
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, conf=0.1, iou=0.5, verbose=True)

    print("\n" + "=" * 50)
    print("Validation Results:")
    for key, val in results.results_dict.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")


def resume_with_modified_config(
    checkpoint_path: Optional[str] = None,
    modifications: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Resume training with modified hyperparameters.

    Args:
        checkpoint_path: Path to checkpoint.
        modifications: Dict of Ultralytics training args to override.

    Returns:
        Training results or *None*.
    """
    if modifications is None:
        modifications = {}

    if checkpoint_path is None:
        print("❌ checkpoint_path is required")
        return None

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    torch.cuda.empty_cache()
    gc.collect()

    defaults = dict(
        epochs=50,
        batch=8,
        imgsz=640,
        device=0,
        optimizer="AdamW",
        lr0=0.0005,
        patience=30,
        cos_lr=True,
        save=True,
        plots=True,
        verbose=True,
    )
    defaults.update(modifications)

    try:
        model = YOLO(checkpoint_path)
        results = model.train(**defaults)
        print("\n✅ Modified resume training completed")
        return results
    except Exception as exc:
        print(f"\n❌ Modified resume failed: {exc}")
        torch.cuda.empty_cache()
        gc.collect()
        return None
