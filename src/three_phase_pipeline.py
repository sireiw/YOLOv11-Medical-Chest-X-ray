"""Integrated three-phase lung disease detection pipeline.

A comprehensive pipeline for lung disease detection that chains:

1. **Phase 1 — YOLO Detection**: Detect lung regions in chest X-rays.
2. **Phase 2 — Segmentation**: Segment the lung using UNet + EfficientNet-B1.
3. **Phase 3 — Classification**: Classify the disease (Normal / TB / Pneumonia / Lung Cancer).

Includes a TB-optimised variant (:class:`TBOptimizedLungPipeline`) with
threshold calibration, ensemble prediction, and dynamic thresholding to
achieve ≥90% recall and ≥70% specificity on Tuberculosis.

Also provides :class:`BatchLungPredictor` for folder-level batch processing
with CSV/JSON/Excel export and visual summary reports.
"""

import datetime
import glob
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
    )
except ImportError:
    classification_report = None
    confusion_matrix = None

from .models import (
    LungClassifier,
    SuppressOutput,
    UNetEfficientNet,
    convert_numpy_types,
)

# Suppress noisy warnings
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ── Configuration ───────────────────────────────────────────────────────────


class PipelineConfig:
    """Configuration for the three-phase pipeline.

    Override any attribute via keyword arguments::

        config = PipelineConfig(YOLO_MODEL_PATH="/my/model.pt", NUM_CLASSES=3)
    """

    # Model paths — update with your trained model paths
    YOLO_MODEL_PATH: str = "/kaggle/input/yolo150/best.pt"
    SEGMENTATION_MODEL_PATH: str = "/kaggle/working/output/best_segmentation_model.pth"
    CLASSIFICATION_MODEL_PATH: str = "/kaggle/working/output/best_classification_model.pth"

    # Image sizes
    SEG_IMAGE_SIZE: int = 256
    CLS_IMAGE_SIZE: int = 224

    # Classification
    NUM_CLASSES: int = 4
    CLASS_NAMES: List[str] = ["Normal", "Tuberculosis", "Pneumonia", "Lung_Cancer"]

    # YOLO thresholds
    YOLO_CONF_THRESHOLD: float = 0.2
    YOLO_IOU_THRESHOLD: float = 0.3

    # Segmentation threshold
    SEG_THRESHOLD: float = 0.5

    # Device
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output
    OUTPUT_DIR: str = "./pipeline_output"

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


# ── Base pipeline ───────────────────────────────────────────────────────────


class ThreePhaseLungPipeline:
    """Three-phase pipeline: YOLO detection → segmentation → classification.

    Args:
        config: :class:`PipelineConfig` instance.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._load_models()
        self._setup_transforms()

    # ── Model loading ───────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load all three models (YOLO, segmentation, classification)."""
        print("Loading models...")

        print("  Loading YOLO model...")
        self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)

        print("  Loading Segmentation model...")
        self.seg_model = UNetEfficientNet()
        self.seg_model.load_state_dict(
            torch.load(self.config.SEGMENTATION_MODEL_PATH,
                        map_location=self.config.DEVICE)
        )
        self.seg_model.to(self.config.DEVICE).eval()

        print("  Loading Classification model...")
        self.cls_model = LungClassifier(num_classes=self.config.NUM_CLASSES)
        self.cls_model.load_state_dict(
            torch.load(self.config.CLASSIFICATION_MODEL_PATH,
                        map_location=self.config.DEVICE)
        )
        self.cls_model.to(self.config.DEVICE).eval()

        print("✓ All models loaded successfully!")

    def _setup_transforms(self) -> None:
        """Set up albumentations transforms for segmentation and classification."""
        if A is None:
            raise ImportError("albumentations is required: pip install albumentations")
        self.seg_transform = A.Compose([
            A.Resize(self.config.SEG_IMAGE_SIZE, self.config.SEG_IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.cls_transform = A.Compose([
            A.Resize(self.config.CLS_IMAGE_SIZE, self.config.CLS_IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # ── Phase methods ───────────────────────────────────────────────────

    def phase1_yolo_detection(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Phase 1: Detect lung region using YOLO.

        Returns:
            ``(image_rgb, detection_info)`` where *detection_info* contains
            ``detected``, ``confidence``, ``bbox``, and ``original_shape``.
        """
        results = self.yolo_model(image_path, conf=self.config.YOLO_CONF_THRESHOLD)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        info: Dict[str, Any] = {
            "detected": False,
            "confidence": 0.0,
            "bbox": None,
            "original_shape": image.shape[:2],
        }

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                best = int(np.argmax(confs))
                x1, y1, x2, y2 = map(int, boxes.xyxy[best].cpu().numpy())
                info["detected"] = True
                info["confidence"] = float(confs[best])
                info["bbox"] = [x1, y1, x2, y2]
                return image_rgb, info

        return image_rgb, info

    def phase2_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Phase 2: Segment lung region.

        Returns:
            ``(segmented_image, binary_mask)``
        """
        augmented = self.seg_transform(image=image)
        tensor = augmented["image"].unsqueeze(0).to(self.config.DEVICE)

        with torch.no_grad():
            logits = self.seg_model(tensor)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

        mask = cv2.resize(prob, (image.shape[1], image.shape[0]))
        mask_bin = (mask > self.config.SEG_THRESHOLD).astype(np.uint8)

        segmented = image.copy()
        segmented[mask_bin == 0] = 0
        return segmented, mask_bin

    def phase3_classification(self, image: np.ndarray) -> Dict:
        """Phase 3: Classify the disease from the segmented image.

        Returns:
            Dict with ``predicted_class``, ``confidence``, ``all_probabilities``.
        """
        augmented = self.cls_transform(image=image)
        tensor = augmented["image"].unsqueeze(0).to(self.config.DEVICE)

        with torch.no_grad():
            logits = self.cls_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return {
            "predicted_class": self.config.CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "all_probabilities": {
                n: float(p) for n, p in zip(self.config.CLASS_NAMES, probs)
            },
        }

    # ── End-to-end ──────────────────────────────────────────────────────

    def process_single_image(
        self, image_path: str, save_intermediate: bool = True
    ) -> Dict:
        """Run all three phases on a single image.

        Args:
            image_path: Path to the chest X-ray image.
            save_intermediate: Whether to save intermediate outputs.

        Returns:
            Dict with ``detection``, ``classification``, and ``intermediate_images``.
        """
        print(f"\nProcessing: {os.path.basename(image_path)}")

        # Phase 1
        print("  Phase 1: YOLO Detection...")
        original, det = self.phase1_yolo_detection(image_path)
        status = (f"✓ Lung detected (conf: {det['confidence']:.3f})"
                  if det["detected"] else "⚠️ No lung region detected")
        print(f"    {status}")

        # Phase 2
        print("  Phase 2: Segmentation...")
        segmented, mask = self.phase2_segmentation(original)
        print("    ✓ Segmentation complete")

        # Phase 3
        print("  Phase 3: Classification...")
        cls_result = self.phase3_classification(segmented)
        print(f"    ✓ {cls_result['predicted_class']} "
              f"(conf: {cls_result['confidence']:.3f})")

        result = {
            "image_path": image_path,
            "detection": det,
            "classification": cls_result,
            "intermediate_images": {
                "original": original,
                "mask": mask,
                "segmented": segmented,
            },
        }

        if save_intermediate:
            self._save_intermediate_results(result)
        return result

    # ── Visualisation ───────────────────────────────────────────────────

    def visualize_pipeline(
        self, image_path: str, save_path: Optional[str] = None
    ) -> None:
        """Create a 2×3 grid visualisation of the pipeline stages."""
        result = self.process_single_image(image_path, save_intermediate=False)
        original_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title("1. Original X-ray", fontsize=14, weight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(original_rgb)
        if result["detection"]["detected"]:
            x1, y1, x2, y2 = result["detection"]["bbox"]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  fill=False, edgecolor="lime", linewidth=3)
            axes[0, 1].add_patch(rect)
            axes[0, 1].set_title(
                f"2. YOLO Detection\n✓ conf: {result['detection']['confidence']:.3f}",
                fontsize=12, color="green")
        else:
            axes[0, 1].set_title("2. YOLO Detection\n✗ No lung detected",
                                  fontsize=12, color="orange")
        axes[0, 1].axis("off")

        axes[0, 2].text(0.5, 0.5,
                         "THREE-PHASE PIPELINE:\n\n1. YOLO Detection\n"
                         "2. Segmentation\n3. Classification\n\n"
                         "All phases use the\nORIGINAL IMAGE",
                         ha="center", va="center", fontsize=12,
                         transform=axes[0, 2].transAxes,
                         bbox=dict(boxstyle="round,pad=0.5",
                                   facecolor="lightblue", alpha=0.7))
        axes[0, 2].axis("off")

        axes[1, 0].imshow(result["intermediate_images"]["mask"], cmap="gray")
        axes[1, 0].set_title("3. Segmentation Mask", fontsize=12, weight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(result["intermediate_images"]["segmented"])
        axes[1, 1].set_title("4. Segmented Lung", fontsize=12, weight="bold")
        axes[1, 1].axis("off")

        cls = result["classification"]
        text = (f"5. Classification Result\n\n"
                f"Prediction: {cls['predicted_class']}\n"
                f"Confidence: {cls['confidence']:.3f}\n\n"
                "All probabilities:\n")
        for name, prob in cls["all_probabilities"].items():
            text += f"{name}: {prob:.3f}\n"
        axes[1, 2].axis("off")
        axes[1, 2].text(0.1, 0.5, text, fontsize=12, verticalalignment="center",
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

        plt.suptitle("Three-Phase Lung Disease Detection Pipeline",
                      fontsize=16, weight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def quick_show(self, image_path: str) -> None:
        """Quick 1×3 visualisation: original → segmented → prediction."""
        result = self.process_single_image(image_path, save_intermediate=False)
        plt.figure(figsize=(12, 5))

        plt.subplot(131)
        plt.imshow(plt.imread(image_path))
        plt.title("Original X-ray")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(result["intermediate_images"]["segmented"])
        plt.title("Segmented Lung")
        plt.axis("off")

        plt.subplot(133)
        plt.text(0.5, 0.5,
                 f"Prediction:\n{result['classification']['predicted_class']}\n\n"
                 f"Confidence:\n{result['classification']['confidence']:.1%}",
                 ha="center", va="center", fontsize=14,
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round", facecolor="lightgreen"))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def create_diagnosis_report(
        self, image_path: str, save_path: Optional[str] = None
    ) -> None:
        """Generate a professional diagnosis report image."""
        result = self.process_single_image(image_path, save_intermediate=False)
        original_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        cls = result["classification"]

        fig = plt.figure(figsize=(11, 8.5))
        plt.figtext(0.5, 0.95, "CHEST X-RAY ANALYSIS REPORT",
                     ha="center", fontsize=20, weight="bold")
        plt.figtext(0.1, 0.88, f"Image: {os.path.basename(image_path)}", fontsize=10)
        plt.figtext(0.1, 0.85,
                     f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M}", fontsize=10)

        det_str = "✓ Lung detected" if result["detection"]["detected"] else "✗ No lung detected"
        det_c = "green" if result["detection"]["detected"] else "orange"
        plt.figtext(0.1, 0.82, f"YOLO Detection: {det_str}",
                     fontsize=10, color=det_c, weight="bold")

        ax1 = plt.subplot2grid((3, 3), (0, 0))
        ax1.imshow(original_rgb)
        ax1.set_title("Original X-ray", fontsize=12)
        ax1.axis("off")

        ax2 = plt.subplot2grid((3, 3), (0, 1))
        ax2.imshow(original_rgb)
        if result["detection"]["detected"]:
            x1, y1, x2, y2 = result["detection"]["bbox"]
            ax2.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         fill=False, edgecolor="lime", linewidth=2))
        ax2.set_title("YOLO Detection", fontsize=12)
        ax2.axis("off")

        ax3 = plt.subplot2grid((3, 3), (0, 2))
        ax3.imshow(result["intermediate_images"]["segmented"])
        ax3.set_title("Segmented Result", fontsize=12)
        ax3.axis("off")

        ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
        ax4.axis("off")
        level = ("High" if cls["confidence"] > 0.8
                 else "Moderate" if cls["confidence"] > 0.6
                 else "Low")
        ax4.text(0.5, 0.7, f"PRIMARY DIAGNOSIS: {cls['predicted_class'].upper()}",
                 fontsize=18, weight="bold", ha="center", transform=ax4.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax4.text(0.5, 0.3,
                 f"Confidence Level: {level} ({cls['confidence']:.1%})",
                 fontsize=14, ha="center", transform=ax4.transAxes)

        ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        classes = list(cls["all_probabilities"].keys())
        probs = list(cls["all_probabilities"].values())
        colors = sns.color_palette("husl", len(classes))
        bars = ax5.bar(classes, probs, color=colors)
        ax5.set_ylabel("Probability", fontsize=12)
        ax5.set_title("Differential Diagnosis Probabilities", fontsize=12)
        ax5.set_ylim(0, 1)
        for bar, p in zip(bars, probs):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{p:.3f}", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    # ── Persistence ─────────────────────────────────────────────────────

    def _save_intermediate_results(self, result: Dict) -> None:
        """Save intermediate images and JSON results to *OUTPUT_DIR*."""
        stem = os.path.splitext(os.path.basename(result["image_path"]))[0]
        out = os.path.join(self.config.OUTPUT_DIR, stem)
        os.makedirs(out, exist_ok=True)

        cv2.imwrite(os.path.join(out, "original.png"),
                     cv2.cvtColor(result["intermediate_images"]["original"],
                                  cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out, "mask.png"),
                     result["intermediate_images"]["mask"] * 255)
        cv2.imwrite(os.path.join(out, "segmented.png"),
                     cv2.cvtColor(result["intermediate_images"]["segmented"],
                                  cv2.COLOR_RGB2BGR))

        save_data = convert_numpy_types({
            "image_path": result["image_path"],
            "detection": result["detection"],
            "classification": result["classification"],
        })
        with open(os.path.join(out, "result.json"), "w") as fh:
            json.dump(save_data, fh, indent=2)


# ── TB-optimised pipeline ──────────────────────────────────────────────────


class TBOptimizedLungPipeline(ThreePhaseLungPipeline):
    """Enhanced pipeline with TB-specific optimisations.

    Target metrics: **≥90% recall** and **≥70% specificity** for Tuberculosis.

    Applies class weighting, probability calibration, ensemble TTA, and
    dynamic thresholding to boost TB detection.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self.tb_threshold: float = 0.15
        self.other_threshold: float = 0.5
        self.use_calibration: bool = True
        self.calibration_factor: float = 1.2
        self.use_ensemble: bool = True
        self.tb_specific_features: bool = True
        self.class_weights = {
            "Normal": 1.0, "Tuberculosis": 1.5,
            "Pneumonia": 1.0, "Lung_Cancer": 1.0,
        }
        print("✓ TB-optimized pipeline initialized!")

    # ── Phase 3 override ────────────────────────────────────────────────

    def phase3_classification(self, image: np.ndarray) -> Dict:
        """TB-optimised classification with calibration and ensemble."""
        augmented = self.cls_transform(image=image)
        tensor = augmented["image"].unsqueeze(0).to(self.config.DEVICE)

        with torch.no_grad():
            logits = self.cls_model(tensor)
            base_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        probs = self._apply_class_weights(base_probs)
        if self.use_calibration:
            probs = self._calibrate_tb_probability(probs)
        if self.use_ensemble:
            probs = self._ensemble_prediction(image, probs)

        pred_class, pred_conf = self._apply_dynamic_threshold(probs)

        if self.tb_specific_features and pred_class == "Tuberculosis":
            feats = self._extract_tb_features(image)
            if self._validate_tb_features(feats):
                pred_conf = min(pred_conf * 1.1, 0.99)

        return {
            "predicted_class": pred_class,
            "confidence": float(pred_conf),
            "all_probabilities": {
                n: float(p) for n, p in zip(self.config.CLASS_NAMES, probs)
            },
            "base_probabilities": {
                n: float(p) for n, p in zip(self.config.CLASS_NAMES, base_probs)
            },
            "optimization_applied": True,
            "tb_threshold_used": self.tb_threshold,
        }

    # ── Optimisation helpers ────────────────────────────────────────────

    def _apply_class_weights(self, probs: np.ndarray) -> np.ndarray:
        w = probs.copy()
        for i, cn in enumerate(self.config.CLASS_NAMES):
            w[i] *= self.class_weights[cn]
        return w / w.sum()

    def _calibrate_tb_probability(self, probs: np.ndarray) -> np.ndarray:
        cal = probs.copy()
        tb_i = self.config.CLASS_NAMES.index("Tuberculosis")
        p = probs[tb_i]

        # Piecewise calibration curve
        if p < 0.1:
            cp = p * 1.5
        elif p < 0.3:
            cp = 0.15 + (p - 0.1) * 1.3
        elif p < 0.5:
            cp = 0.41 + (p - 0.3) * 1.2
        else:
            cp = 0.65 + (p - 0.5) * 0.7
        cp = min(cp, 0.99)
        cal[tb_i] = cp

        others = [i for i in range(len(probs)) if i != tb_i]
        remaining = 1.0 - cp
        other_sum = sum(probs[i] for i in others)
        if remaining > 0 and other_sum > 0:
            for i in others:
                cal[i] = probs[i] * remaining / other_sum
        return cal

    def _ensemble_prediction(
        self, image: np.ndarray, base_probs: np.ndarray
    ) -> np.ndarray:
        augmentations = [
            lambda x: x,
            lambda x: cv2.flip(x, 1),
            lambda x: np.clip(x * 1.1, 0, 255).astype(np.uint8),
            lambda x: np.clip(x * 0.9, 0, 255).astype(np.uint8),
            lambda x: np.clip(
                x + np.random.normal(0, 0.01 * 255, x.shape), 0, 255
            ).astype(np.uint8),
        ]

        all_probs = [base_probs]
        for fn in augmentations[1:]:
            aug_img = fn(image)
            aug = self.cls_transform(image=aug_img)
            t = aug["image"].unsqueeze(0).to(self.config.DEVICE)
            with torch.no_grad():
                logits = self.cls_model(t)
                p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                all_probs.append(self._apply_class_weights(p))

        tb_i = self.config.CLASS_NAMES.index("Tuberculosis")
        weights = np.array([1.0 + p[tb_i] * 0.5 for p in all_probs])
        weights /= weights.sum()

        ensemble = np.zeros_like(base_probs)
        for p, w in zip(all_probs, weights):
            ensemble += p * w
        return ensemble

    def _apply_dynamic_threshold(self, probs: np.ndarray) -> Tuple[str, float]:
        tb_i = self.config.CLASS_NAMES.index("Tuberculosis")
        tb_p = probs[tb_i]
        if tb_p >= self.tb_threshold:
            max_other = max(probs[i] for i in range(len(probs)) if i != tb_i)
            if tb_p > max_other * 0.8 or tb_p >= self.tb_threshold * 1.2:
                return "Tuberculosis", tb_p
        idx = int(np.argmax(probs))
        return self.config.CLASS_NAMES[idx], probs[idx]

    def _extract_tb_features(self, image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        return {
            "upper_lobe_mean": float(np.mean(gray[: h // 3, :])),
            "upper_lobe_std": float(np.std(gray[: h // 3, :])),
            "texture_energy": float(np.sum(gray ** 2) / (h * w)),
            "cavity_count": self._count_cavities(gray),
            "asymmetry": float(abs(np.mean(gray[:, : w // 2]) - np.mean(gray[:, w // 2 :]))),
        }

    @staticmethod
    def _count_cavities(gray: np.ndarray) -> int:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for c in contours if 100 < cv2.contourArea(c) < 5000)

    @staticmethod
    def _validate_tb_features(features: Dict) -> bool:
        score = (
            (features["upper_lobe_std"] > 30)
            + (features["texture_energy"] > 100)
            + (features["cavity_count"] > 2)
            + (features["asymmetry"] > 10)
        )
        return score >= 2

    # ── Threshold optimisation ──────────────────────────────────────────

    def optimize_thresholds(
        self,
        validation_data: List[Tuple[str, str]],
        target_recall: float = 0.9,
        target_specificity: float = 0.7,
    ) -> Dict:
        """Sweep TB thresholds on validation data to find the optimal one.

        Args:
            validation_data: List of ``(image_path, label)`` pairs.
            target_recall: Target recall for TB.
            target_specificity: Target specificity for TB.

        Returns:
            Dict with best threshold, achieved recall, and specificity.
        """
        if confusion_matrix is None:
            raise ImportError("scikit-learn is required for threshold optimisation")

        results = []
        for thresh in np.arange(0.05, 0.5, 0.05):
            self.tb_threshold = thresh
            y_true, y_pred = [], []
            for img, label in tqdm(validation_data, desc=f"thresh={thresh:.2f}"):
                with SuppressOutput():
                    res = self.process_single_image(img, save_intermediate=False)
                y_true.append(1 if label == "Tuberculosis" else 0)
                y_pred.append(1 if res["classification"]["predicted_class"] == "Tuberculosis" else 0)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            rec = tp / (tp + fn) if (tp + fn) else 0
            spec = tn / (tn + fp) if (tn + fp) else 0
            dist = np.sqrt((rec - target_recall) ** 2 + (spec - target_specificity) ** 2)
            results.append({"threshold": thresh, "recall": rec, "specificity": spec, "distance": dist})
            print(f"  thresh={thresh:.2f}: R={rec:.3f} S={spec:.3f}")

        best = min(results, key=lambda x: x["distance"])
        self.tb_threshold = best["threshold"]
        print(f"\n✓ Optimal threshold: {self.tb_threshold:.2f} "
              f"(R={best['recall']:.3f}, S={best['specificity']:.3f})")
        return best


# ── Batch predictor ─────────────────────────────────────────────────────────


class BatchLungPredictor:
    """Batch processor for running the pipeline on a folder of images.

    Args:
        pipeline: A :class:`ThreePhaseLungPipeline` or subclass.
        verbose: Print per-image errors.
    """

    SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

    def __init__(
        self,
        pipeline: Union[ThreePhaseLungPipeline, TBOptimizedLungPipeline],
        verbose: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.verbose = verbose

    def predict_folder(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        save_individual_reports: bool = False,
        save_visualizations: bool = False,
        save_intermediate: bool = False,
        recursive: bool = False,
        max_images: Optional[int] = None,
    ) -> pd.DataFrame:
        """Process all images in a folder and export results.

        Generates CSV, JSON, Excel, and a visual summary report.

        Args:
            folder_path: Directory containing images.
            output_dir: Where to save outputs (auto-timestamped if *None*).
            save_individual_reports: Save per-image diagnosis reports.
            save_visualizations: Save per-image pipeline visualisations.
            save_intermediate: Save intermediate phase images.
            recursive: Search sub-directories recursively.
            max_images: Cap on the number of images to process.

        Returns:
            :class:`pd.DataFrame` with prediction results.
        """
        if output_dir is None:
            output_dir = f"predictions_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        os.makedirs(output_dir, exist_ok=True)

        images = self._get_image_paths(folder_path, recursive)
        if not images:
            print(f"No images found in {folder_path}")
            return pd.DataFrame()
        if max_images:
            images = images[:max_images]

        is_tb = isinstance(self.pipeline, TBOptimizedLungPipeline)
        print(f"\n{'=' * 60}\nBATCH PROCESSING STARTED"
              f"{' (TB-OPTIMIZED)' if is_tb else ''}\n{'=' * 60}")
        print(f"Images: {len(images)} | Output: {output_dir}\n")

        results, failed = [], []
        for img in tqdm(images, desc="Processing", ncols=100):
            try:
                with SuppressOutput():
                    res = self.pipeline.process_single_image(
                        img, save_intermediate=save_intermediate)
                row = {
                    "image_path": img,
                    "filename": os.path.basename(img),
                    "yolo_detected": res["detection"]["detected"],
                    "yolo_confidence": res["detection"]["confidence"],
                    "predicted_class": res["classification"]["predicted_class"],
                    "prediction_confidence": res["classification"]["confidence"],
                }
                for cn, p in res["classification"]["all_probabilities"].items():
                    row[f"prob_{cn.lower()}"] = p
                if is_tb and "tb_threshold_used" in res["classification"]:
                    row["tb_optimized"] = True
                    row["tb_threshold"] = res["classification"]["tb_threshold_used"]
                results.append(row)

                if save_individual_reports:
                    rp = os.path.join(output_dir, "reports",
                                      f"{Path(img).stem}_report.png")
                    os.makedirs(os.path.dirname(rp), exist_ok=True)
                    with SuppressOutput():
                        self.pipeline.create_diagnosis_report(img, rp)
                if save_visualizations:
                    vp = os.path.join(output_dir, "visualizations",
                                      f"{Path(img).stem}_pipeline.png")
                    os.makedirs(os.path.dirname(vp), exist_ok=True)
                    with SuppressOutput():
                        self.pipeline.visualize_pipeline(img, vp)

            except Exception as exc:
                failed.append({"image_path": img, "error": str(exc)})
                if self.verbose:
                    print(f"\nError: {img}: {exc}")

        df = pd.DataFrame(results)
        self._save_results(df, failed, output_dir)
        self._print_summary(df, failed, output_dir, is_tb)
        return df

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_image_paths(self, folder: str, recursive: bool) -> List[str]:
        paths: List[str] = []
        for ext in self.SUPPORTED_FORMATS:
            pattern = os.path.join(folder, "**" if recursive else "", f"*{ext}")
            paths.extend(glob.glob(pattern, recursive=recursive))
        return sorted(set(paths))

    def _save_results(
        self, df: pd.DataFrame, failed: List[Dict], output_dir: str
    ) -> None:
        df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        summary = {
            "total_processed": int(len(df)),
            "total_failed": int(len(failed)),
            "processing_date": datetime.datetime.now().isoformat(),
        }
        if len(df) > 0:
            summary["class_distribution"] = {
                k: int(v) for k, v in df["predicted_class"].value_counts().items()
            }
            summary["average_confidence"] = float(df["prediction_confidence"].mean())

        json_data = {"summary": summary, "predictions": df.to_dict("records"),
                      "failed_images": failed}
        with open(os.path.join(output_dir, "predictions_detailed.json"), "w") as f:
            json.dump(json_data, f, indent=2, default=str)

    @staticmethod
    def _print_summary(
        df: pd.DataFrame, failed: List[Dict], output_dir: str, is_tb: bool
    ) -> None:
        n = len(df) + len(failed)
        print(f"\n{'=' * 60}\nPROCESSING COMPLETE\n{'=' * 60}")
        print(f"✓ Processed: {len(df)} | ✗ Failed: {len(failed)} "
              f"| Rate: {len(df) / n * 100:.1f}%")
        print(f"📁 Results → {output_dir}/")
        if len(df) > 0:
            print("\nDisease Distribution:")
            for disease, count in df["predicted_class"].value_counts().items():
                print(f"  {disease}: {count} ({count / len(df) * 100:.1f}%)")
        print()
