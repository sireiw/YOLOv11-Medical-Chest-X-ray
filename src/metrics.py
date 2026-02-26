"""Recall and specificity analysis for YOLO detection models.

Provides functions for evaluating model performance on class-level
recall and specificity, with optional verbose output control.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ultralytics import YOLO


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def analyze_recall_and_specificity(
    model_path: str = "/kaggle/input/yolo150/best.pt",
    data_yaml: str = "/kaggle/working/yolov11_dataset_1k_guaranteed/dataset.yaml",
    conf_threshold: float = 0.10,
    iou_threshold: float = 0.50,
    verbose: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Analyse per-class recall and specificity.

    Args:
        model_path: Path to trained weights.
        data_yaml: Dataset YAML config.
        conf_threshold: Confidence threshold.
        iou_threshold: IoU threshold.
        verbose: Whether to print Ultralytics output.

    Returns:
        ``(results_df, metrics_dict)`` or ``(None, None)`` on failure.
    """
    try:
        model = YOLO(model_path)

        if verbose:
            results = model.val(
                data=data_yaml, conf=conf_threshold, iou=iou_threshold,
            )
        else:
            with suppress_stdout():
                results = model.val(
                    data=data_yaml, conf=conf_threshold, iou=iou_threshold,
                    verbose=False,
                )

        # Extract metrics
        class_names = ["Lung_Cancer", "Pneumonia", "Tuberculosis"]
        metrics_data = []

        for i, cls_name in enumerate(class_names):
            try:
                precision = float(results.box.p[i]) if i < len(results.box.p) else 0
                recall = float(results.box.r[i]) if i < len(results.box.r) else 0
                ap50 = float(results.box.ap50[i]) if i < len(results.box.ap50) else 0
                # Specificity approximation: TN / (TN + FP) ≈ 1 - FP_rate
                specificity = max(0, 1.0 - (1.0 - precision) * recall / (precision + 1e-6))

                metrics_data.append({
                    "Class": cls_name,
                    "Precision": precision,
                    "Recall": recall,
                    "Specificity": specificity,
                    "AP@50": ap50,
                    "F1": 2 * precision * recall / (precision + recall + 1e-6),
                })
            except (IndexError, AttributeError):
                metrics_data.append({"Class": cls_name, "Precision": 0, "Recall": 0,
                                     "Specificity": 0, "AP@50": 0, "F1": 0})

        df = pd.DataFrame(metrics_data)

        # Overall metrics
        overall = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "mean_precision": df["Precision"].mean(),
            "mean_recall": df["Recall"].mean(),
            "mean_specificity": df["Specificity"].mean(),
            "mean_f1": df["F1"].mean(),
        }

        # Print results
        print("\n" + "=" * 70)
        print("  Recall & Specificity Analysis")
        print("=" * 70)
        print(f"\n  Model: {model_path}")
        print(f"  Conf threshold: {conf_threshold}")
        print(f"  IoU threshold: {iou_threshold}")
        print(f"\n{df.to_string(index=False, float_format='{:.4f}'.format)}")
        print(f"\n  mAP@50: {overall['mAP50']:.4f}")
        print(f"  mAP@50-95: {overall['mAP50-95']:.4f}")
        print(f"  Mean F1: {overall['mean_f1']:.4f}")
        print("=" * 70)

        return df, overall

    except Exception as exc:
        print(f"Error during analysis: {exc}")
        return None, None
