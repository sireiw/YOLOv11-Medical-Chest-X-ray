"""Dataset diagnostics for YOLO label quality.

Functions for diagnosing class visibility issues, analysing bounding-box
size distributions, checking annotation quality, and suggesting
improvements.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["Lung_Cancer", "Pneumonia", "Tuberculosis"]


def diagnose_class_visibility(
    dataset_path: str = "/kaggle/working/yolov11_dataset_1k_guaranteed",
) -> None:
    """Check how visible each class's annotations are.

    Compares average bbox sizes across classes and highlights classes
    whose boxes may be too small for the model to detect.

    Args:
        dataset_path: Root of the YOLO dataset.
    """
    dataset_path = Path(dataset_path)
    class_sizes: Dict[str, List[float]] = {c: [] for c in CLASS_NAMES}

    for split in ("train", "val", "test"):
        label_dir = dataset_path / split / "labels"
        if not label_dir.exists():
            continue
        for lf in label_dir.glob("*.txt"):
            with open(lf) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    area = w * h
                    class_sizes[CLASS_NAMES[cls_id]].append(area)

    print("=== Class Visibility Diagnosis ===\n")
    for cls_name, areas in class_sizes.items():
        if not areas:
            print(f"  {cls_name}: No annotations found ❌")
            continue
        arr = np.array(areas)
        print(f"  {cls_name}:")
        print(f"    Count      : {len(arr)}")
        print(f"    Mean area  : {arr.mean():.6f}")
        print(f"    Median area: {np.median(arr):.6f}")
        print(f"    Min area   : {arr.min():.6f}")
        print(f"    Max area   : {arr.max():.6f}")
        if arr.mean() < 0.01:
            print(f"    ⚠️  Very small boxes — model may struggle to detect")
        else:
            print(f"    ✓  Reasonable box sizes")
        print()

    # Side-by-side comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [class_sizes[c] for c in CLASS_NAMES if class_sizes[c]]
    labels = [c for c in CLASS_NAMES if class_sizes[c]]
    ax.boxplot(data, labels=labels)
    ax.set_title("Bounding-Box Area Distribution by Class")
    ax.set_ylabel("Normalised Area (w × h)")
    plt.tight_layout()
    plt.savefig("class_visibility_comparison.png", dpi=150)
    plt.show()


def analyze_bbox_sizes(
    dataset_path: str = "/kaggle/working/yolov11_dataset_1k_guaranteed",
) -> None:
    """Plot detailed bbox size histograms and cumulative distributions.

    Args:
        dataset_path: Root of the YOLO dataset.
    """
    dataset_path = Path(dataset_path)
    all_areas: List[float] = []
    class_areas: Dict[str, List[float]] = {c: [] for c in CLASS_NAMES}

    for split in ("train", "val", "test"):
        label_dir = dataset_path / split / "labels"
        if not label_dir.exists():
            continue
        for lf in label_dir.glob("*.txt"):
            with open(lf) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    area = w * h
                    all_areas.append(area)
                    class_areas[CLASS_NAMES[cls_id]].append(area)

    if not all_areas:
        print("No bounding boxes found")
        return

    colors = ["red", "green", "blue"]

    # Histogram
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(6 * len(CLASS_NAMES), 5))
    for i, cls_name in enumerate(CLASS_NAMES):
        if class_areas[cls_name]:
            axes[i].hist(class_areas[cls_name], bins=50, color=colors[i], alpha=0.7)
        axes[i].set_title(f"{cls_name} BBox Areas")
        axes[i].set_xlabel("Normalised Area")
        axes[i].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("bbox_size_distribution.png", dpi=150)
    plt.show()

    # Cumulative
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cls_name in enumerate(CLASS_NAMES):
        if not class_areas[cls_name]:
            continue
        sorted_areas = np.sort(class_areas[cls_name])
        cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
        ax.plot(sorted_areas, cumulative, label=cls_name, color=colors[i], linewidth=2)
    ax.set_title("Cumulative BBox Area Distribution")
    ax.set_xlabel("Normalised Area")
    ax.set_ylabel("Cumulative Proportion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bbox_cumulative_distribution.png", dpi=150)
    plt.show()


def check_annotation_quality(
    dataset_path: str = "/kaggle/working/yolov11_dataset_1k_guaranteed",
) -> Dict[str, int]:
    """Check for common annotation issues.

    Returns:
        Dictionary of issue type → count.
    """
    dataset_path = Path(dataset_path)
    issues: Dict[str, int] = {
        "tiny_boxes": 0,
        "huge_boxes": 0,
        "extreme_aspect_ratio": 0,
        "edge_boxes": 0,
        "total_boxes": 0,
    }

    for split in ("train", "val", "test"):
        label_dir = dataset_path / split / "labels"
        if not label_dir.exists():
            continue
        for lf in label_dir.glob("*.txt"):
            with open(lf) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cx, cy, w, h = map(float, parts[1:])
                    area = w * h
                    issues["total_boxes"] += 1

                    if area < 0.001:
                        issues["tiny_boxes"] += 1
                    if area > 0.5:
                        issues["huge_boxes"] += 1
                    ar = max(w / (h + 1e-6), h / (w + 1e-6))
                    if ar > 5:
                        issues["extreme_aspect_ratio"] += 1
                    if cx - w / 2 < 0.02 or cx + w / 2 > 0.98:
                        issues["edge_boxes"] += 1

    print("=== Annotation Quality Check ===\n")
    for key, val in issues.items():
        flag = "⚠️" if val > 0 and key != "total_boxes" else "✓"
        print(f"  {flag} {key}: {val}")
    return issues


def suggest_solutions(
    dataset_path: str = "/kaggle/working/yolov11_dataset_1k_guaranteed",
) -> None:
    """Run diagnostics and print suggested training adjustments."""
    issues = check_annotation_quality(dataset_path)
    print("\n=== Suggested Solutions ===\n")

    if issues["tiny_boxes"] > 0:
        print("• Tiny boxes detected:")
        print("  → Use imgsz=1024 or higher")
        print("  → Lower conf_threshold to 0.05")
        print("  → Add targeted augmentation for small objects")

    if issues["huge_boxes"] > 0:
        print("• Very large boxes detected:")
        print("  → Verify annotations are correct")
        print("  → Consider mosaic augmentation")

    if issues["extreme_aspect_ratio"] > 0:
        print("• Extreme aspect ratios:")
        print("  → Verify bbox coordinates")
        print("  → Add aspect-ratio-aware augmentation")

    if issues["edge_boxes"] > 0:
        print("• Edge boxes detected:")
        print("  → Apply translate augmentation = 0.2")
        print("  → Use letterbox padding consistently")

    if all(issues[k] == 0 for k in ("tiny_boxes", "huge_boxes", "extreme_aspect_ratio")):
        print("✓ No significant annotation issues detected.")
