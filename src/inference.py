"""High-recall inference pipeline for YOLOv11 detection.

Provides :class:`HighRecallInference` for running predictions with low
confidence thresholds and Test-Time Augmentation (TTA) to maximise recall.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO


class HighRecallInference:
    """Inference pipeline optimised for high-recall TB detection.

    Uses low confidence thresholds and optional TTA to catch as many
    positive cases as possible, even at the cost of some precision.

    Args:
        model_path: Path to trained ``.pt`` weights.
        conf_threshold: Confidence threshold (lower → higher recall).
        iou_threshold: IoU threshold for NMS (lower → more detections).
    """

    CLASS_NAMES = ["Lung_Cancer", "Pneumonia", "Tuberculosis"]

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.3,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {device} | conf={conf_threshold} iou={iou_threshold}")

    # ── Single image ────────────────────────────────────────────────────

    def process_single_image(
        self,
        image_path: str,
        save_dir: Optional[str] = None,
        visualize: bool = True,
    ) -> Dict[str, Any]:
        """Process one image with high-recall settings.

        Args:
            image_path: Path to the image file.
            save_dir: Optional directory to save annotated results.
            visualize: Whether to display results inline.

        Returns:
            Dictionary with detection results.
        """
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        detections: List[Dict[str, Any]] = []
        for box in boxes:
            det = {
                "class_id": int(box.cls[0]),
                "class_name": self.CLASS_NAMES[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            }
            detections.append(det)

        if visualize:
            self._visualize(image_path, detections)

        if save_dir:
            self._save_results(image_path, detections, save_dir)

        return {
            "image_path": image_path,
            "num_detections": len(detections),
            "detections": detections,
        }

    # ── Folder processing ───────────────────────────────────────────────

    def process_folder(
        self,
        test_folder: str,
        output_dir: str = "test_results",
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> List[Dict[str, Any]]:
        """Process every image in a folder.

        Args:
            test_folder: Path to the folder of test images.
            output_dir: Where to save annotated outputs.
            image_extensions: Image file extensions to include.

        Returns:
            List of per-image result dictionaries.
        """
        test_path = Path(test_folder)
        images = [
            f for f in test_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not images:
            print(f"No images found in {test_folder}")
            return []

        os.makedirs(output_dir, exist_ok=True)
        all_results: List[Dict[str, Any]] = []

        for img_path in tqdm(images, desc="Processing images"):
            res = self.process_single_image(
                str(img_path), save_dir=output_dir, visualize=False,
            )
            all_results.append(res)

        # Summary
        detection_summary = self._summarize(all_results)
        self.create_summary_report(all_results, detection_summary, output_dir)

        return all_results

    # ── TTA ─────────────────────────────────────────────────────────────

    def apply_tta_inference(
        self,
        image_path: str,
        augmentations: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Apply Test-Time Augmentation for improved recall.

        Args:
            image_path: Path to the image.
            augmentations: List of augmentations to apply
                (``'original'``, ``'flip'``, ``'brightness'``).

        Returns:
            Combined list of detections after TTA.
        """
        if augmentations is None:
            augmentations = ["original", "flip", "brightness"]

        img = cv2.imread(image_path)
        if img is None:
            return []

        all_detections: List[Dict[str, Any]] = []
        for aug in augmentations:
            if aug == "original":
                aug_img = img
            elif aug == "flip":
                aug_img = cv2.flip(img, 1)
            elif aug == "brightness":
                aug_img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
            else:
                continue

            results = self.model(
                aug_img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            boxes = self._extract_boxes(results[0])
            all_detections.extend(boxes)

        return all_detections

    # ── Report ──────────────────────────────────────────────────────────

    def create_summary_report(
        self,
        results: List[Dict[str, Any]],
        detection_summary: Dict[str, int],
        output_dir: str,
    ) -> None:
        """Write a summary report of all detections to *output_dir*."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(results),
            "images_with_detections": sum(
                1 for r in results if r["num_detections"] > 0
            ),
            "total_detections": sum(r["num_detections"] for r in results),
            "class_distribution": detection_summary,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }

        path = Path(output_dir) / "summary_report.json"
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSummary report saved to {path}")

        # Print
        print(f"\nTotal images: {report['total_images']}")
        print(f"Images with detections: {report['images_with_detections']}")
        print(f"Total detections: {report['total_detections']}")
        for cls_name, count in detection_summary.items():
            print(f"  {cls_name}: {count}")

    # ── Private helpers ─────────────────────────────────────────────────

    def _visualize(self, image_path: str, detections: List[Dict]) -> None:
        """Display an image with bounding-box overlays."""
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = colors[det["class_id"] % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"{Path(image_path).name} — {len(detections)} detections")
        plt.axis("off")
        plt.show()

    def _save_results(
        self, image_path: str, detections: List[Dict], save_dir: str
    ) -> None:
        """Save annotated image and JSON results."""
        img = cv2.imread(image_path)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = colors[det["class_id"] % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        stem = Path(image_path).stem
        cv2.imwrite(str(Path(save_dir) / f"{stem}_annotated.png"), img)

        with open(Path(save_dir) / f"{stem}_detections.json", "w") as fh:
            json.dump(detections, fh, indent=2)

    def _extract_boxes(self, result: Any) -> List[Dict[str, Any]]:
        """Extract box dicts from an Ultralytics result object."""
        boxes = []
        for box in result.boxes:
            boxes.append({
                "class_id": int(box.cls[0]),
                "class_name": self.CLASS_NAMES[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            })
        return boxes

    @staticmethod
    def _summarize(results: List[Dict]) -> Dict[str, int]:
        """Count detections per class across all results."""
        counts: Dict[str, int] = {}
        for r in results:
            for det in r["detections"]:
                name = det["class_name"]
                counts[name] = counts.get(name, 0) + 1
        return counts
