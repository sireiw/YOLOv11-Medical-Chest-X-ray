"""YOLO detection result display and annotation utilities.

Functions for visualising model predictions on individual images and
batches, with options for saving annotated outputs.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = ["Lung_Cancer", "Pneumonia", "Tuberculosis"]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def display_with_yolo_plot(
    model_path: str,
    image_path: str,
    conf: float = 0.10,
    iou: float = 0.30,
) -> None:
    """Display detections using Ultralytics built-in ``plot()`` method."""
    model = YOLO(model_path)
    results = model(image_path, conf=conf, iou=iou, verbose=False)
    annotated = results[0].plot()

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(f"YOLOv11 Detections (conf≥{conf})")
    plt.axis("off")
    plt.show()


def display_custom_bbox(
    model_path: str,
    image_path: str,
    conf: float = 0.10,
    iou: float = 0.30,
    show_confidence: bool = True,
    bbox_thickness: int = 2,
    font_scale: float = 0.7,
) -> None:
    """Display detections with custom-styled bounding boxes.

    Args:
        model_path: Path to ``.pt`` model weights.
        image_path: Path to the image.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        show_confidence: Whether to annotate with confidence values.
        bbox_thickness: Bounding-box line thickness.
        font_scale: Font scale for labels.
    """
    model = YOLO(model_path)
    results = model(image_path, conf=conf, iou=iou, verbose=False)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        color = COLORS[cls_id % len(COLORS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, bbox_thickness)
        label = CLASS_NAMES[cls_id]
        if show_confidence:
            label += f" {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.title(f"Custom BBox Display (conf≥{conf})")
    plt.axis("off")
    plt.show()


def display_before_after(
    model_path: str,
    image_path: str,
    conf: float = 0.10,
    iou: float = 0.30,
) -> None:
    """Side-by-side: original image vs. annotated detections."""
    model = YOLO(model_path)
    results = model(image_path, conf=conf, iou=iou, verbose=False)

    original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(annotated)
    axes[1].set_title(f"Detections (conf≥{conf})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def display_batch_results(
    model_path: str,
    image_folder: str,
    conf: float = 0.10,
    iou: float = 0.30,
    max_images: int = 6,
    cols: int = 3,
) -> None:
    """Display detection results for multiple images in a grid.

    Args:
        model_path: Model weights path.
        image_folder: Folder containing test images.
        conf: Confidence threshold.
        iou: IoU threshold.
        max_images: Maximum number of images to show.
        cols: Number of grid columns.
    """
    model = YOLO(model_path)
    folder = Path(image_folder)
    images = sorted(folder.glob("*.png"))[:max_images]
    if not images:
        print(f"No images found in {image_folder}")
        return

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(images):
        ax = axes[idx // cols, idx % cols]
        results = model(str(img_path), conf=conf, iou=iou, verbose=False)
        annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        ax.imshow(annotated)
        n_det = len(results[0].boxes)
        ax.set_title(f"{img_path.name}\n{n_det} detections", fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for idx in range(len(images), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(f"Batch Results (conf≥{conf}, iou≥{iou})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def save_annotated_image(
    model_path: str,
    image_path: str,
    output_dir: str = "annotated_results",
    conf: float = 0.10,
    iou: float = 0.30,
) -> str:
    """Run detection and save the annotated image.

    Returns:
        Path to the saved annotated image.
    """
    model = YOLO(model_path)
    results = model(image_path, conf=conf, iou=iou, verbose=False)
    annotated = results[0].plot()

    os.makedirs(output_dir, exist_ok=True)
    out_path = str(Path(output_dir) / f"annotated_{Path(image_path).name}")
    cv2.imwrite(out_path, annotated)
    print(f"Saved: {out_path}")
    return out_path


def quick_display(model_path: str, image_path: str) -> None:
    """One-liner: load model → detect → show."""
    display_with_yolo_plot(model_path, image_path)
