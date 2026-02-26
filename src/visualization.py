"""Visualisation utilities for YOLO-format medical imaging datasets.

Provides helpers for displaying annotated samples, per-class statistics,
and detailed single-image breakdowns inside Jupyter notebooks.
"""

import random
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES

# Colour palette for the three classes
_COLORS = ["red", "green", "blue"]


# ── Sample display ──────────────────────────────────────────────────────────


def display_yolo_dataset(
    dataset_path: str,
    num_samples: int = 6,
    samples_per_row: int = 3,
    class_names: Optional[List[str]] = None,
) -> None:
    """Show random annotated samples from each split.

    Args:
        dataset_path: Root of the YOLO dataset directory.
        num_samples: How many images to sample per split.
        samples_per_row: Grid columns.
        class_names: Override default class names.
    """
    dataset_path = Path(dataset_path)
    cls_names = class_names or CLASS_NAMES
    colors = _COLORS

    for split in ("train", "val", "test"):
        img_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"
        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*.png"))
        if not images:
            continue

        sample = random.sample(images, min(num_samples, len(images)))
        num_rows = (len(sample) + samples_per_row - 1) // samples_per_row

        fig, axes = plt.subplots(
            num_rows, samples_per_row,
            figsize=(5 * samples_per_row, 5 * num_rows),
        )

        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif samples_per_row == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(
            f"{split.upper()} Split — Sample Images with Annotations",
            fontsize=16, fontweight="bold",
        )

        for idx, img_path in enumerate(sample):
            ax = axes[idx // samples_per_row, idx % samples_per_row]
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            _draw_boxes(ax, label_dir / f"{img_path.stem}.txt", img, cls_names, colors)
            ax.set_title(f"{img_path.stem[:20]}…", fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for idx in range(len(sample), num_rows * samples_per_row):
            axes[idx // samples_per_row, idx % samples_per_row].axis("off")

        plt.tight_layout()
        plt.show()
        print(f"Displayed {len(sample)} samples from {split} split\n")


# ── Statistics ──────────────────────────────────────────────────────────────


def display_statistics(
    dataset_path: str,
    class_names: Optional[List[str]] = None,
) -> None:
    """Bar charts showing split distribution, class counts, and average boxes.

    Args:
        dataset_path: Root of the YOLO dataset directory.
        class_names: Override default class names.
    """
    dataset_path = Path(dataset_path)
    cls_names = class_names or CLASS_NAMES

    split_counts = {}
    boxes_per_class = {c: 0 for c in cls_names}
    images_with_class = {c: 0 for c in cls_names}

    for split in ("train", "val", "test"):
        img_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"
        if not img_dir.exists():
            split_counts[split] = 0
            continue
        split_counts[split] = len(list(img_dir.glob("*.png")))

        for lf in label_dir.glob("*.txt"):
            seen: set = set()
            with open(lf) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cidx = int(parts[0])
                    cn = cls_names[cidx]
                    boxes_per_class[cn] += 1
                    seen.add(cn)
            for cn in seen:
                images_with_class[cn] += 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Dataset Statistics", fontsize=16, fontweight="bold")

    # 1 — split distribution
    ax = axes[0, 0]
    ax.bar(split_counts.keys(), split_counts.values(), color=["blue", "orange", "green"])
    ax.set_title("Images per Split")
    ax.set_ylabel("Number of Images")
    for i, (_, v) in enumerate(split_counts.items()):
        ax.text(i, v + max(split_counts.values()) * 0.01, str(v), ha="center", fontweight="bold")

    # 2 — images per class
    ax = axes[0, 1]
    vals = [images_with_class[c] for c in cls_names]
    ax.bar(range(len(cls_names)), vals, color=_COLORS)
    ax.set_xticks(range(len(cls_names)))
    ax.set_xticklabels(cls_names, rotation=45, ha="right")
    ax.set_title("Images Containing Each Class")
    ax.set_ylabel("Number of Images")

    # 3 — total boxes per class
    ax = axes[1, 0]
    vals = [boxes_per_class[c] for c in cls_names]
    ax.bar(range(len(cls_names)), vals, color=_COLORS)
    ax.set_xticks(range(len(cls_names)))
    ax.set_xticklabels(cls_names, rotation=45, ha="right")
    ax.set_title("Total Bounding Boxes per Class")
    ax.set_ylabel("Number of Boxes")

    # 4 — average boxes per image
    ax = axes[1, 1]
    avgs = [
        boxes_per_class[c] / images_with_class[c] if images_with_class[c] else 0
        for c in cls_names
    ]
    ax.bar(range(len(cls_names)), avgs, color=_COLORS)
    ax.set_xticks(range(len(cls_names)))
    ax.set_xticklabels(cls_names, rotation=45, ha="right")
    ax.set_title("Average Boxes per Image")
    ax.set_ylabel("Average Number of Boxes")

    plt.tight_layout()
    plt.show()

    print(f"\n=== Dataset Summary ===")
    print(f"Total images: {sum(split_counts.values())}")
    print(f"Total bounding boxes: {sum(boxes_per_class.values())}")
    print("\nPer-class breakdown:")
    for cn in cls_names:
        print(f"  {cn}: {images_with_class[cn]} images, {boxes_per_class[cn]} boxes")


# ── Detailed single-image view ──────────────────────────────────────────────


def display_sample_with_details(
    dataset_path: str,
    image_name: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """Display one image with annotations and a textual breakdown.

    Args:
        dataset_path: Root of the YOLO dataset directory.
        image_name: Specific filename to show; random if *None*.
        class_names: Override default class names.
    """
    dataset_path = Path(dataset_path)
    cls_names = class_names or CLASS_NAMES
    colors = _COLORS

    # Locate the image
    img_path = _find_image(dataset_path, image_name)
    if img_path is None:
        print(f"Image {'not found' if image_name else 'directory empty'}")
        return

    split = img_path.parent.parent.name
    label_path = dataset_path / split / "labels" / f"{img_path.stem}.txt"

    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(15, 5))

    # Panel 1 — original
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Panel 2 — annotated
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(img_rgb)
    boxes_info = _draw_boxes(
        ax2, label_path, img, cls_names, colors, numbered=True
    )
    ax2.set_title("Annotated Image")
    ax2.axis("off")

    # Panel 3 — text details
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis("off")
    text = (
        f"Image: {img_path.name}\n"
        f"Split: {split}\n"
        f"Dimensions: {img.shape[1]}x{img.shape[0]}\n"
        f"Total boxes: {len(boxes_info)}\n\n"
    )
    if boxes_info:
        text += "Bounding Boxes:\n"
        for i, b in enumerate(boxes_info, 1):
            text += (
                f"\nBox {i}: {b['class']}\n"
                f"  Center: ({b['center'][0]:.3f}, {b['center'][1]:.3f})\n"
                f"  Size: {b['size'][0]:.3f} x {b['size'][1]:.3f}\n"
                f"  Area: {b['area']:.4f}\n"
            )
    ax3.text(
        0.1, 0.9, text,
        transform=ax3.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
    )
    ax3.set_title("Annotation Details")

    plt.tight_layout()
    plt.show()


# ── Internal helpers ────────────────────────────────────────────────────────


def _find_image(dataset_path: Path, image_name: Optional[str]) -> Optional[Path]:
    """Locate an image by name, or pick a random one from 'train'."""
    if image_name is None:
        img_dir = dataset_path / "train" / "images"
        images = list(img_dir.glob("*.png"))
        return random.choice(images) if images else None

    for split in ("train", "val", "test"):
        candidate = dataset_path / split / "images" / image_name
        if candidate.exists():
            return candidate
    return None


def _draw_boxes(
    ax, label_path: Path, img: np.ndarray,
    cls_names: List[str], colors: List[str],
    numbered: bool = False,
) -> list:
    """Draw YOLO bboxes on *ax* and return a list of box-info dicts."""
    info: list = []
    if not label_path.exists():
        return info

    h, w = img.shape[:2]
    with open(label_path) as fh:
        for i, line in enumerate(fh):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cidx = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            pw, ph = bw * w, bh * h

            rect = patches.Rectangle(
                (x1, y1), pw, ph,
                linewidth=2, edgecolor=colors[cidx], facecolor="none",
            )
            ax.add_patch(rect)

            label = cls_names[cidx]
            if numbered:
                label = f"{label} #{i + 1}"
            ax.text(
                x1, y1 - 5, label,
                color=colors[cidx], fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

            info.append({
                "class": cls_names[cidx],
                "center": (cx, cy),
                "size": (bw, bh),
                "area": bw * bh,
            })
    return info
