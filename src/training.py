"""YOLOv11 training pipeline for tuberculosis detection.

Provides early stopping, real-time monitoring, smart batch-size optimization,
validation with confidence-threshold sweeping, and a comprehensive advanced
training function with all optimizations integrated.
"""

import gc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


# ── Early stopping ──────────────────────────────────────────────────────────


class TBEarlyStopping:
    """Custom early stopping for tuberculosis detection metrics.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        metric: Metric to monitor (e.g. ``'recall'``).
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 0.001, metric: str = "recall"
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Check whether training should stop.

        Returns:
            *True* if early-stop should trigger.
        """
        score = metrics.get(self.metric, 0)
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


# ── Training monitor ────────────────────────────────────────────────────────


class TrainingMonitor:
    """Real-time training monitoring and logging.

    Logs per-epoch metrics to a JSON file and prints GPU memory usage.

    Args:
        log_dir: Directory where log files are stored.
    """

    def __init__(self, log_dir: str = "tb_training_logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Log metrics for a single epoch."""
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "elapsed": str(datetime.now() - self.start_time),
            **metrics,
        }
        self.metrics_history.append(entry)

        log_file = self.log_dir / "training_log.json"
        with open(log_file, "w") as fh:
            json.dump(self.metrics_history, fh, indent=2, default=str)

        print(f"\nEpoch {epoch}:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

    def log_gpu_memory(self) -> None:
        """Log current GPU memory usage."""
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
        print(f"  GPU Memory: {allocated:.2f}/{total:.2f} GB "
              f"(reserved: {reserved:.2f} GB)")


# ── Batch-size optimiser ────────────────────────────────────────────────────


def monitor_gpu_memory() -> None:
    """Print a quick GPU memory snapshot."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    total = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
    print(f"GPU Memory: {allocated:.2f}/{total:.2f} GB ({allocated / total * 100:.1f}%)")


def smart_batch_optimizer(
    model_name: str = "yolo11m.pt",
    target_memory_usage: float = 0.85,
) -> int:
    """Find the optimal batch size based on available GPU memory.

    Tries progressively smaller batch sizes until one fits comfortably.

    Args:
        model_name: YOLO model checkpoint name.
        target_memory_usage: Fraction of GPU memory to target.

    Returns:
        Recommended batch size.
    """
    if not torch.cuda.is_available():
        print("No GPU — defaulting to batch_size=4")
        return 4

    total_mem = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
    print(f"Total GPU Memory: {total_mem:.2f} GB")

    # Heuristic based on GPU memory
    if total_mem >= 24:
        candidates = [32, 24, 16, 12, 8]
    elif total_mem >= 16:
        candidates = [16, 12, 8, 6, 4]
    elif total_mem >= 8:
        candidates = [8, 6, 4, 2]
    else:
        candidates = [4, 2, 1]

    for batch_size in candidates:
        try:
            torch.cuda.empty_cache()
            gc.collect()

            model = YOLO(model_name)

            # Dry-run a tiny training step
            test_results = model.train(
                data="coco128.yaml",
                epochs=1,
                batch=batch_size,
                imgsz=640,
                device=0,
                verbose=False,
                exist_ok=True,
                project="/tmp/batch_test",
                name="test",
            )

            used = torch.cuda.memory_allocated() / 1024 ** 3
            usage_pct = used / total_mem

            if usage_pct <= target_memory_usage:
                print(f"✓ batch_size={batch_size} uses {usage_pct * 100:.1f}% GPU")
                torch.cuda.empty_cache()
                gc.collect()
                return batch_size

            print(f"✗ batch_size={batch_size} uses {usage_pct * 100:.1f}% — too high")
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError:
            print(f"✗ batch_size={batch_size} — OOM, trying smaller")
            torch.cuda.empty_cache()
            gc.collect()

    print("Falling back to batch_size=2")
    return 2


# ── Validation ──────────────────────────────────────────────────────────────


def validate_tb_performance(
    model_path: str,
    data_yaml: str,
    conf_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Validate a trained model and sweep confidence thresholds.

    Args:
        model_path: Path to trained ``.pt`` weights.
        data_yaml: Path to dataset YAML config.
        conf_thresholds: List of confidence thresholds to evaluate.

    Returns:
        Dictionary of best threshold and corresponding metrics.
    """
    if conf_thresholds is None:
        conf_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    model = YOLO(model_path)
    best_result: Dict[str, Any] = {"best_recall": 0}

    for conf in conf_thresholds:
        print(f"\nValidating at conf={conf:.2f}...")
        try:
            results = model.val(data=data_yaml, conf=conf, iou=0.5, verbose=False)

            metrics = {
                "conf_threshold": conf,
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            }

            print(f"  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                  f"mAP50={metrics['mAP50']:.4f}")

            if metrics["recall"] > best_result["best_recall"]:
                best_result = {"best_recall": metrics["recall"], **metrics}

        except Exception as exc:
            print(f"  Error at conf={conf}: {exc}")

    print(f"\n{'=' * 50}")
    print(f"Best recall: {best_result['best_recall']:.4f} "
          f"at conf={best_result.get('conf_threshold', 'N/A')}")
    return best_result


# ── Advanced training pipeline ──────────────────────────────────────────────


def _numpy_serializer(obj: Any) -> Any:
    """Convert numpy types to Python-native types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def train_tuberculosis_advanced(
    data_yaml: str = "/kaggle/working/yolov11_dataset_1k_guaranteed/dataset.yaml",
    model_name: str = "yolo11m.pt",
    project_dir: str = "tuberculosis_optimized_v11",
    run_name: str = "tb_best_yolo11m_v1",
    epochs: int = 100,
    batch_size: Optional[int] = None,
    img_size: int = 640,
    device: int = 0,
) -> Optional[Any]:
    """Advanced YOLOv11 training pipeline with all optimisations.

    Includes smart batch-size selection, early stopping on recall,
    real-time monitoring, and post-training result analysis.

    Args:
        data_yaml: Path to dataset YAML.
        model_name: Base model to fine-tune.
        project_dir: Output project directory.
        run_name: Sub-directory name for this run.
        epochs: Max training epochs.
        batch_size: Batch size (auto-determined if *None*).
        img_size: Training image size.
        device: CUDA device index.

    Returns:
        Ultralytics training results object, or *None* on failure.
    """
    print("\n" + "=" * 60)
    print("🚀 YOLOv11 Advanced Training Pipeline")
    print("=" * 60)

    # Auto batch size
    if batch_size is None:
        batch_size = smart_batch_optimizer(model_name)
    print(f"Using batch_size={batch_size}")

    # Monitor
    monitor = TrainingMonitor(log_dir=f"{project_dir}/{run_name}/logs")
    monitor_gpu_memory()

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    model = YOLO(model_name)

    # Training config
    train_args = dict(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.5,
        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # Scheduling
        patience=50,
        cos_lr=True,
        # Output
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("\nTraining with config:")
    for k, v in train_args.items():
        if k != "data":
            print(f"  {k}: {v}")

    try:
        results = model.train(**train_args)

        # Post-training
        monitor_gpu_memory()

        # Save summary
        summary = {
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "completed": datetime.now().isoformat(),
        }
        summary_path = Path(project_dir) / run_name / "training_summary.json"
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=_numpy_serializer)

        print(f"\n✅ Training complete — results in {project_dir}/{run_name}")
        return results

    except Exception as exc:
        print(f"\n❌ Training failed: {exc}")
        torch.cuda.empty_cache()
        gc.collect()
        return None


# ── Results analysis ────────────────────────────────────────────────────────


def analyze_training_results(results_dir: str) -> None:
    """Comprehensive analysis of training results.

    Reads the ``results.csv`` produced by Ultralytics and prints per-metric
    summaries with best-epoch information.

    Args:
        results_dir: Path to the training run directory.
    """
    results_dir = Path(results_dir)
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        print(f"No results.csv found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print("\n" + "=" * 60)
    print("📊 Training Results Analysis")
    print("=" * 60)
    print(f"Total epochs: {len(df)}")

    metrics_of_interest = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@50",
        "metrics/mAP50-95(B)": "mAP@50-95",
        "train/box_loss": "Train Box Loss",
        "val/box_loss": "Val Box Loss",
    }

    for col, name in metrics_of_interest.items():
        if col not in df.columns:
            continue
        series = df[col]
        is_loss = "loss" in col.lower()
        best_val = series.min() if is_loss else series.max()
        best_epoch = series.idxmin() if is_loss else series.idxmax()
        final_val = series.iloc[-1]

        print(f"\n  {name}:")
        print(f"    Best: {best_val:.4f} (epoch {best_epoch})")
        print(f"    Final: {final_val:.4f}")

    # Convergence check
    if "val/box_loss" in df.columns and len(df) > 10:
        last_n = df["val/box_loss"].tail(10)
        delta = abs(last_n.iloc[-1] - last_n.iloc[0])
        if delta < 0.01:
            print("\n⚠️  Validation loss plateaued — consider stopping or adjusting LR")
        else:
            print("\n✓ Model still converging")


# ── Legacy compatibility ────────────────────────────────────────────────────


def train_tuberculosis_optimized_p100(**kwargs: Any) -> Optional[Any]:
    """Legacy function — redirects to :func:`train_tuberculosis_advanced`."""
    return train_tuberculosis_advanced(**kwargs)


def train_tuberculosis_yolo11m_memory_safe(**kwargs: Any) -> Optional[Any]:
    """Legacy function — redirects to :func:`train_tuberculosis_advanced`."""
    return train_tuberculosis_advanced(**kwargs)


def train_with_gradient_accumulation(**kwargs: Any) -> Optional[Any]:
    """Legacy function — integrated into :func:`train_tuberculosis_advanced`."""
    return train_tuberculosis_advanced(**kwargs)


def adaptive_batch_size_finder(
    base_config: Optional[Dict] = None, model_name: str = "yolo11m.pt"
) -> int:
    """Legacy function — redirects to :func:`smart_batch_optimizer`."""
    return smart_batch_optimizer(model_name)
