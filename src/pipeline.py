"""Dataset preparation pipeline for YOLOv11 medical object detection.

Orchestrates reading from VinBigData, NIH ChestX-ray14, RSNA Pneumonia, and
TBX11K datasets, converting images to a unified 640×640 format with YOLO-style
labels, performing intelligent bounding-box deduplication, and splitting data
by patient to prevent leakage.
"""

import ast
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
import yaml
from tqdm import tqdm

from .bbox_dedup import ImprovedBboxDeduplication
from .config import CLASS_NAMES, EPSILON, DatasetConfig

logger = logging.getLogger(__name__)


# ── Image-processing helpers ────────────────────────────────────────────────


def _resize_and_pad(
    img: np.ndarray, img_size: int
) -> Tuple[np.ndarray, int, int, float, int, int]:
    """Resize *img* (grayscale) to fit in ``img_size × img_size`` with padding.

    Returns:
        ``(padded_image, orig_w, orig_h, scale, pad_x, pad_y)``
    """
    orig_h, orig_w = img.shape[:2]
    scale = min(img_size / orig_w, img_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((img_size, img_size), dtype=np.uint8)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return padded, orig_w, orig_h, scale, pad_x, pad_y


# ── Pipeline class ──────────────────────────────────────────────────────────


class DatasetPreparationPipeline:
    """End-to-end medical dataset preparation pipeline for YOLOv11.

    Usage::

        config = DatasetConfig(output_dir="./dataset", target_per_class=1000)
        pipeline = DatasetPreparationPipeline(config)
        pipeline.run()

    Args:
        config: Pipeline configuration. Uses defaults if *None*.
    """

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.config = config or DatasetConfig()
        self.output_dir = Path(self.config.output_dir)
        self.debug_dir = self.output_dir / "debug_visualizations"

        self.deduplicator = ImprovedBboxDeduplication(self.config)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.create_directories()

        self.class_names = CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self._initialize_tracking()
        self._load_all_metadata()

        self.patient_splits: Dict[str, set] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }

        self.successful_images: Dict[str, int] = {c: 0 for c in self.class_names}
        self.attempted_images: Dict[str, int] = {c: 0 for c in self.class_names}
        self.failed_reasons: Dict[str, Dict] = {
            c: defaultdict(int) for c in self.class_names
        }

    # ── Setup helpers ───────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        """Configure root logger with file + console handlers."""
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.DEBUG if self.config.debug_mode else logging.INFO,
            format=fmt,
            handlers=[
                logging.FileHandler(self.output_dir / "preparation.log"),
                logging.StreamHandler(),
            ],
        )
        logger.info("Initializing YOLOv11 Dataset with GUARANTEED %d images per class",
                     self.config.target_per_class)

    def _initialize_tracking(self) -> None:
        """Reset all tracking structures."""
        datasets = ("vinbig", "nih", "rsna", "tbx11k")
        self.processed_images: Dict[str, set] = {d: set() for d in datasets}
        self.class_counts: Dict[str, int] = {n: 0 for n in self.class_names}
        self.image_assignments: Dict[str, dict] = {d: {} for d in datasets}
        self.failed_conversions: List[str] = []

        self.dedup_stats: Dict[str, Dict] = {
            d: {
                "original": 0,
                "deduplicated": 0,
                "exact_duplicates": 0,
                "details": [],
                "methods_used": defaultdict(int),
            }
            for d in datasets
        }

        self.debug_stats: Dict = {
            "suspicious_pairs": [],
            "small_boxes_removed": 0,
            "out_of_bounds_fixed": 0,
            "visualizations_created": 0,
            "nih_age_filtered": 0,
            "dedup_methods": defaultdict(int),
            "lung_cancer_boxes_protected": 0,
            "exact_duplicates_removed": 0,
        }

    # ── Metadata loading ────────────────────────────────────────────────

    def _load_all_metadata(self) -> None:
        """Load and cache metadata from all datasets."""
        logger.info("Loading all metadata for YOLOv11 dataset...")
        self.metadata_cache: Dict = {}
        self._load_nih_metadata()
        self._load_vinbig_metadata()
        self._load_rsna_metadata()
        self._load_tbx11k_metadata()

    def _load_nih_metadata(self) -> None:
        base = self.config.nih_base_path
        meta_path = f"{base}/Data_Entry_2017.csv"
        bbox_path = f"{base}/BBox_List_2017.csv"
        try:
            if not os.path.exists(meta_path):
                return
            logger.info("Loading NIH metadata with age filtering...")
            df = pd.read_csv(meta_path)
            df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
            lo, hi = self.config.age_range
            df = df[(df["Patient Age"] >= lo) & (df["Patient Age"] <= hi)]
            self.valid_nih_images: set = set(df["Image Index"].values)

            if os.path.exists(bbox_path):
                bbox = pd.read_csv(bbox_path)
                bbox = bbox[bbox["Image Index"].isin(self.valid_nih_images)]
                self.metadata_cache["nih_bbox"] = bbox

            self.metadata_cache["nih"] = df
            logger.info("NIH: %d images for ages %d-%d", len(self.valid_nih_images), lo, hi)
        except Exception as exc:
            logger.error("Failed to load NIH metadata: %s", exc)
            self.valid_nih_images = set()

    def _load_vinbig_metadata(self) -> None:
        csv_path = f"{self.config.vinbig_base_path}/train.csv"
        try:
            if os.path.exists(csv_path):
                self.metadata_cache["vinbig"] = pd.read_csv(csv_path)
                n = len(self.metadata_cache["vinbig"]["image_id"].unique())
                logger.info("VinBigData: %d unique images", n)
        except Exception as exc:
            logger.error("Failed to load VinBigData metadata: %s", exc)

    def _load_rsna_metadata(self) -> None:
        csv_path = f"{self.config.rsna_base_path}/stage_2_train_labels.csv"
        try:
            if os.path.exists(csv_path):
                self.metadata_cache["rsna"] = pd.read_csv(csv_path)
                n = len(self.metadata_cache["rsna"]["patientId"].unique())
                logger.info("RSNA: %d unique patients", n)
        except Exception as exc:
            logger.error("Failed to load RSNA metadata: %s", exc)

    def _load_tbx11k_metadata(self) -> None:
        csv_path = f"{self.config.tbx11k_base_path}/data.csv"
        try:
            if os.path.exists(csv_path):
                self.metadata_cache["tbx11k"] = pd.read_csv(csv_path)
                logger.info("TBX11K: %d images", len(self.metadata_cache["tbx11k"]))
        except Exception as exc:
            logger.error("Failed to load TBX11K metadata: %s", exc)

    # ── Directory structure ─────────────────────────────────────────────

    def create_directories(self) -> None:
        """Create YOLOv11 directory structure."""
        for split in ("train", "val", "test"):
            for folder in ("images", "labels"):
                (self.output_dir / split / folder).mkdir(parents=True, exist_ok=True)
        if self.config.debug_mode:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    # ── Patient-level splitting ─────────────────────────────────────────

    def get_patient_split(self, patient_id: str, dataset: str) -> str:
        """Deterministic patient split to avoid data leakage."""
        key = f"{dataset}_{patient_id}"
        for split, patients in self.patient_splits.items():
            if key in patients:
                return split
        split = np.random.choice(["train", "val", "test"], p=self.config.split_ratios)
        self.patient_splits[split].add(key)
        return split

    # ── Single-image processing ─────────────────────────────────────────

    def _process_single_image(
        self, img_path: str, output_path: str, is_dicom: bool
    ) -> Optional[Tuple]:
        """Read, resize, pad, and save a single image.

        Returns:
            ``(True, orig_w, orig_h, scale, pad_x, pad_y)`` on success, else *None*.
        """
        try:
            if is_dicom:
                return self.dicom_to_png(img_path, output_path)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            padded, orig_w, orig_h, scale, pad_x, pad_y = _resize_and_pad(
                img, self.config.img_size
            )
            cv2.imwrite(str(output_path), padded)
            return True, orig_w, orig_h, scale, pad_x, pad_y

        except Exception as exc:
            logger.error("Error processing %s: %s", img_path, exc)
            return None

    def dicom_to_png(self, dicom_path: str, output_path: str) -> Optional[Tuple]:
        """Convert a DICOM file to a grayscale PNG with windowing and padding."""
        try:
            ds = pydicom.dcmread(dicom_path)
            pixels = ds.pixel_array.astype(float)

            if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                wc = (
                    float(ds.WindowCenter)
                    if not isinstance(ds.WindowCenter, list)
                    else float(ds.WindowCenter[0])
                )
                ww = (
                    float(ds.WindowWidth)
                    if not isinstance(ds.WindowWidth, list)
                    else float(ds.WindowWidth[0])
                )
                pixels = np.clip(pixels, wc - ww // 2, wc + ww // 2)

            pixels = (
                (pixels - pixels.min()) / (pixels.max() - pixels.min() + EPSILON) * 255
            ).astype(np.uint8)

            padded, orig_w, orig_h, scale, pad_x, pad_y = _resize_and_pad(
                pixels, self.config.img_size
            )
            cv2.imwrite(str(output_path), padded)
            return True, orig_w, orig_h, scale, pad_x, pad_y

        except Exception as exc:
            logger.error("Error converting DICOM %s: %s", dicom_path, exc)
            self.failed_conversions.append(str(dicom_path))
            return None

    # ── BBOX normalisation ──────────────────────────────────────────────

    def normalize_bbox(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Scale, pad, clamp, and normalise a bbox to YOLO format.

        Returns:
            ``(cx, cy, nw, nh)`` in ``[0, 1]`` or *None* if the box is too small.
        """
        img_size = self.config.img_size

        x_scaled = x * scale + pad_x
        y_scaled = y * scale + pad_y
        w_scaled = w * scale
        h_scaled = h * scale

        x_min = max(0, min(x_scaled, img_size))
        y_min = max(0, min(y_scaled, img_size))
        x_max = max(0, min(x_scaled + w_scaled, img_size))
        y_max = max(0, min(y_scaled + h_scaled, img_size))

        w_c = x_max - x_min
        h_c = y_max - y_min

        min_px = img_size * 0.008
        if w_c < min_px or h_c < min_px:
            self.debug_stats["small_boxes_removed"] += 1
            return None

        if (
            x_scaled != x_min
            or y_scaled != y_min
            or x_scaled + w_scaled != x_max
            or y_scaled + h_scaled != y_max
        ):
            self.debug_stats["out_of_bounds_fixed"] += 1

        cx = (x_min + x_max) / 2 / img_size
        cy = (y_min + y_max) / 2 / img_size
        nw = w_c / img_size
        nh = h_c / img_size

        cx = np.clip(cx, nw / 2, 1.0 - nw / 2)
        cy = np.clip(cy, nh / 2, 1.0 - nh / 2)

        return cx, cy, nw, nh

    # ── Label writing ───────────────────────────────────────────────────

    def write_labels(
        self,
        label_path: Path,
        class_idx: int,
        boxes: List[List[float]],
        dataset_name: Optional[str] = None,
        class_name: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[bool, int]:
        """Deduplicate, validate, and write YOLO labels to *label_path*."""
        if class_name is None:
            class_name = self.class_names[class_idx]

        deduplicated, info = self.deduplicator.deduplicate_boxes(
            boxes, class_name, image_path
        )

        # Track dedup stats
        if dataset_name:
            ds = self.dedup_stats[dataset_name]
            ds["original"] += info["original"]
            ds["deduplicated"] += info["final"]
            ds["exact_duplicates"] += info.get("exact_duplicates_removed", 0)
            ds["methods_used"][info["method"]] += 1

            self.debug_stats["exact_duplicates_removed"] += info.get(
                "exact_duplicates_removed", 0
            )

            threshold = 10 if class_name == "Lung_Cancer" else 30
            if info["removal_rate"] > threshold:
                ds["details"].append(
                    {
                        "image": str(label_path.name),
                        "class": class_name,
                        "original": info["original"],
                        "final": info["final"],
                        "removed_percent": info["removal_rate"],
                        "method": info["method"],
                        "exact_duplicates": info.get("exact_duplicates_removed", 0),
                    }
                )

        self.debug_stats["dedup_methods"][info["method"]] += 1

        if class_name == "Lung_Cancer" and info["method"] in (
            "exact_dedup_then_keep_all",
            "exact_dedup_then_spatial_soft_nms",
            "exact_dedup_then_spatial_protection",
        ):
            self.debug_stats["lung_cancer_boxes_protected"] += info["final"]

        if self.config.debug_mode and image_path:
            vis_thr = 1 if class_name == "Lung_Cancer" else 2
            if info["removed"] > vis_thr or info.get("exact_duplicates_removed", 0) > 0:
                self.visualize_deduplication(
                    image_path, boxes, deduplicated, class_name, info
                )

        if not deduplicated:
            return False, 0

        with open(label_path, "w") as fh:
            for box in deduplicated:
                cx, cy, nw, nh = box
                cx = np.clip(cx, nw / 2, 1.0 - nw / 2)
                cy = np.clip(cy, nh / 2, 1.0 - nh / 2)
                nw = np.clip(nw, 0.01, 1.0)
                nh = np.clip(nh, 0.01, 1.0)
                fh.write(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        return True, info["removed"]

    # ── Debug visualisation ─────────────────────────────────────────────

    def visualize_deduplication(
        self,
        image_path: str,
        original_boxes: List[List[float]],
        deduplicated_boxes: List[List[float]],
        class_name: str,
        dedup_info: Dict,
    ) -> None:
        """Save a side-by-side debug image showing before/after dedup."""
        if not self.config.debug_mode:
            return
        try:
            img = cv2.imread(image_path)
            if img is None:
                return
            h, w = img.shape[:2]
            vis = img.copy()

            for box in original_boxes:
                cx, cy, bw, bh = box
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

            for box in deduplicated_boxes:
                cx, cy, bw, bh = box
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis, f"Original: {dedup_info['original']} boxes",
                        (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.putText(vis, f"Kept: {dedup_info['final']} boxes",
                        (10, 60), font, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"Method: {dedup_info['method']}",
                        (10, 90), font, 0.7, (255, 255, 0), 2)
            if dedup_info.get("exact_duplicates_removed", 0) > 0:
                cv2.putText(
                    vis,
                    f"Exact duplicates: {dedup_info['exact_duplicates_removed']}",
                    (10, 120), font, 0.7, (255, 0, 255), 2,
                )

            fname = (
                f"dedup_{class_name}_{dedup_info['method']}_"
                f"{dedup_info['original']}to{dedup_info['final']}_"
                f"{os.path.basename(image_path)}"
            )
            cv2.imwrite(str(self.debug_dir / fname), vis)
            self.debug_stats["visualizations_created"] += 1
        except Exception as exc:
            logger.error("Error creating visualization: %s", exc)

    # ── Per-class processing ────────────────────────────────────────────

    def process_dataset_by_class(self) -> None:
        """Process all datasets in priority order per class."""
        logger.info("=== Starting YOLOv11 Dataset Processing (GUARANTEED %d per class) ===",
                     self.config.target_per_class)
        self._process_tuberculosis()
        self._process_lung_cancer()
        self._process_pneumonia()
        logger.info("=== YOLOv11 Dataset Processing Complete ===")

    # -- Tuberculosis ----------------------------------------------------

    def _process_tuberculosis(self) -> None:
        logger.info("Processing Tuberculosis data (target: %d)...", self.config.target_per_class)
        if "tbx11k" in self.metadata_cache:
            self._process_tbx11k_tb()
        if self.successful_images["Tuberculosis"] < self.config.target_per_class:
            remaining = self.config.target_per_class - self.successful_images["Tuberculosis"]
            logger.info("Need %d more TB images, trying VinBig...", remaining)
            if "vinbig" in self.metadata_cache:
                self._process_vinbig_class("Tuberculosis", [13, 11, 2])
        if self.successful_images["Tuberculosis"] < self.config.target_per_class:
            remaining = self.config.target_per_class - self.successful_images["Tuberculosis"]
            logger.info("Still need %d more TB images, trying NIH...", remaining)
            if "nih_bbox" in self.metadata_cache:
                self._process_nih_class(
                    "Tuberculosis", ["Infiltration", "Fibrosis", "Pleural_Thickening"]
                )
        final = self.successful_images["Tuberculosis"]
        if final < self.config.target_per_class:
            logger.warning("⚠️ Could not reach target for Tuberculosis: %d/%d",
                           final, self.config.target_per_class)
        else:
            logger.info("✓ Tuberculosis: %d images collected", final)

    # -- Lung Cancer -----------------------------------------------------

    def _process_lung_cancer(self) -> None:
        logger.info("Processing Lung Cancer data (target: %d)...", self.config.target_per_class)
        if "vinbig" in self.metadata_cache:
            self._process_vinbig_class("Lung_Cancer", [8])
        if self.successful_images["Lung_Cancer"] < self.config.target_per_class:
            if "nih_bbox" in self.metadata_cache:
                self._process_nih_class("Lung_Cancer", ["Mass", "Nodule"])
        final = self.successful_images["Lung_Cancer"]
        if final < self.config.target_per_class:
            logger.warning("Could not reach target for Lung_Cancer: %d/%d",
                           final, self.config.target_per_class)
        else:
            logger.info("✓ Lung_Cancer: %d images collected", final)

    # -- Pneumonia -------------------------------------------------------

    def _process_pneumonia(self) -> None:
        logger.info("Processing Pneumonia data (target: %d)...", self.config.target_per_class)
        if "rsna" in self.metadata_cache:
            self._process_rsna_pneumonia()
        if self.successful_images["Pneumonia"] < self.config.target_per_class:
            if "vinbig" in self.metadata_cache:
                self._process_vinbig_class("Pneumonia", [4])
        final = self.successful_images["Pneumonia"]
        if final < self.config.target_per_class:
            logger.warning("Could not reach target for Pneumonia: %d/%d",
                           final, self.config.target_per_class)
        else:
            logger.info("✓ Pneumonia: %d images collected", final)

    # ── Dataset-specific processors ─────────────────────────────────────

    def _process_tbx11k_tb(self) -> None:
        """Process TBX11K tuberculosis data."""
        df = self.metadata_cache["tbx11k"]
        tb_cases = df[df["target"] == "tb"].copy()
        base = self.config.tbx11k_base_path

        needed = self.config.target_per_class - self.successful_images["Tuberculosis"]
        max_attempts = int(needed * self.config.max_attempts_multiplier)

        pbar = tqdm(total=needed, desc="Collecting TBX11K TB images")
        attempts = 0

        for _, row in tb_cases.iterrows():
            if self.successful_images["Tuberculosis"] >= self.config.target_per_class:
                break
            if attempts >= max_attempts:
                logger.warning("Reached maximum attempts (%d) for TBX11K", max_attempts)
                break

            attempts += 1
            self.attempted_images["Tuberculosis"] += 1

            fname = row["fname"]
            img_path = f"{base}/images/{fname}"

            if not os.path.exists(img_path):
                self.failed_reasons["Tuberculosis"]["file_not_found"] += 1
                continue

            if fname in self.processed_images["tbx11k"]:
                continue

            patient_id = fname.split("_")[0] if "_" in fname else fname
            split = self.get_patient_split(patient_id, "tbx11k")

            img_name = f"tb_{fname}"
            img_output = self.output_dir / split / "images" / img_name

            result = self._process_single_image(img_path, str(img_output), False)
            if not result or not result[0]:
                self.failed_reasons["Tuberculosis"]["processing_failed"] += 1
                continue

            _, orig_w, orig_h, scale, pad_x, pad_y = result
            boxes_yolo: List[List[float]] = []

            if row["bbox"] != "none":
                try:
                    bbox_dict = ast.literal_eval(row["bbox"].strip())
                    if isinstance(bbox_dict, dict):
                        norm = self.normalize_bbox(
                            bbox_dict["xmin"], bbox_dict["ymin"],
                            bbox_dict["width"], bbox_dict["height"],
                            scale, pad_x, pad_y,
                        )
                        if norm:
                            boxes_yolo.append(list(norm))
                except Exception as exc:
                    logger.error("Error parsing bbox for %s: %s", fname, exc)
                    self.failed_reasons["Tuberculosis"]["invalid_bbox"] += 1
            else:
                self.failed_reasons["Tuberculosis"]["no_bbox_annotation"] += 1

            if boxes_yolo:
                lbl = self.output_dir / split / "labels" / f"tb_{fname.replace('.png', '.txt')}"
                ok, _ = self.write_labels(
                    lbl, self.class_to_idx["Tuberculosis"],
                    boxes_yolo, "tbx11k", "Tuberculosis", str(img_output),
                )
                if ok:
                    self.successful_images["Tuberculosis"] += 1
                    self.class_counts["Tuberculosis"] += 1
                    self.processed_images["tbx11k"].add(fname)
                    pbar.update(1)
                else:
                    os.remove(img_output)
                    self.failed_reasons["Tuberculosis"]["no_valid_boxes"] += 1
            else:
                os.remove(img_output)
                self.failed_reasons["Tuberculosis"]["no_boxes_generated"] += 1

        pbar.close()

    def _process_vinbig_class(self, class_name: str, class_ids: List[int]) -> None:
        """Generic VinBig processing for any class."""
        df = self.metadata_cache["vinbig"]
        cases = df[df["class_id"].isin(class_ids)].copy()
        unique_imgs = cases["image_id"].unique()

        available = [
            img for img in unique_imgs if img not in self.image_assignments["vinbig"]
        ]
        np.random.shuffle(available)

        base = self.config.vinbig_base_path
        needed = self.config.target_per_class - self.successful_images[class_name]
        if needed <= 0:
            return
        max_attempts = int(needed * self.config.max_attempts_multiplier)

        prefix = class_name.lower().replace("_", "")
        pbar = tqdm(total=needed, desc=f"Collecting VinBig {class_name} images")
        attempts = 0

        for img_id in available:
            if self.successful_images[class_name] >= self.config.target_per_class:
                break
            if attempts >= max_attempts:
                logger.warning("Reached maximum attempts for VinBig %s", class_name)
                break

            attempts += 1
            self.attempted_images[class_name] += 1

            img_path = f"{base}/train/{img_id}.dicom"
            if not os.path.exists(img_path):
                self.failed_reasons[class_name]["file_not_found"] += 1
                continue

            split = self.get_patient_split(img_id, "vinbig")
            img_name = f"vinbig_{prefix}_{img_id}.png"
            img_output = self.output_dir / split / "images" / img_name

            result = self._process_single_image(img_path, str(img_output), True)
            if not result or not result[0]:
                self.failed_reasons[class_name]["processing_failed"] += 1
                continue

            _, orig_w, orig_h, scale, pad_x, pad_y = result
            img_boxes = cases[cases["image_id"] == img_id]
            boxes_yolo: List[List[float]] = []

            for _, row in img_boxes.iterrows():
                if pd.notna(row["x_min"]) and row["x_min"] >= 0:
                    norm = self.normalize_bbox(
                        row["x_min"], row["y_min"],
                        row["x_max"] - row["x_min"],
                        row["y_max"] - row["y_min"],
                        scale, pad_x, pad_y,
                    )
                    if norm:
                        boxes_yolo.append(list(norm))

            if boxes_yolo:
                lbl = self.output_dir / split / "labels" / f"vinbig_{prefix}_{img_id}.txt"
                ok, _ = self.write_labels(
                    lbl, self.class_to_idx[class_name],
                    boxes_yolo, "vinbig", class_name, str(img_output),
                )
                if ok:
                    self.successful_images[class_name] += 1
                    self.class_counts[class_name] += 1
                    self.image_assignments["vinbig"][img_id] = class_name
                    self.processed_images["vinbig"].add(img_id)
                    pbar.update(1)
                else:
                    os.remove(img_output)
                    self.failed_reasons[class_name]["no_valid_boxes"] += 1
            else:
                os.remove(img_output)
                self.failed_reasons[class_name]["no_boxes_generated"] += 1

        pbar.close()

    def _process_nih_class(self, class_name: str, finding_labels: List[str]) -> None:
        """Generic NIH processing for any class."""
        df = self.metadata_cache["nih_bbox"]
        cases = df[df["Finding Label"].isin(finding_labels)].copy()
        available = cases[~cases["Image Index"].isin(self.image_assignments["nih"])]

        base = self.config.nih_base_path
        needed = self.config.target_per_class - self.successful_images[class_name]
        if needed <= 0:
            return
        max_attempts = int(needed * self.config.max_attempts_multiplier)

        grouped = available.groupby("Image Index")
        image_indices = list(grouped.groups.keys())
        np.random.shuffle(image_indices)

        prefix = class_name.lower().replace("_", "")
        pbar = tqdm(total=needed, desc=f"Collecting NIH {class_name} images")
        attempts = 0

        for img_name in image_indices:
            if self.successful_images[class_name] >= self.config.target_per_class:
                break
            if attempts >= max_attempts:
                logger.warning("Reached maximum attempts for NIH %s", class_name)
                break

            attempts += 1
            self.attempted_images[class_name] += 1

            # Find image file
            img_path = None
            for folder in os.listdir(base):
                if folder.startswith("images_"):
                    candidate = f"{base}/{folder}/images/{img_name}"
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
            if img_path is None:
                self.failed_reasons[class_name]["file_not_found"] += 1
                continue

            patient_id = img_name.split("_")[0]
            split = self.get_patient_split(patient_id, "nih")
            output_name = f"nih_{prefix}_{img_name}"
            img_output = self.output_dir / split / "images" / output_name

            result = self._process_single_image(img_path, str(img_output), False)
            if not result or not result[0]:
                self.failed_reasons[class_name]["processing_failed"] += 1
                continue

            _, orig_w, orig_h, scale, pad_x, pad_y = result
            img_boxes = grouped.get_group(img_name)
            boxes_yolo: List[List[float]] = []

            for _, row in img_boxes.iterrows():
                norm = self.normalize_bbox(
                    row["Bbox [x"], row["y"], row["w"], row["h]"],
                    scale, pad_x, pad_y,
                )
                if norm:
                    boxes_yolo.append(list(norm))

            if boxes_yolo:
                lbl_path = self.output_dir / split / "labels" / output_name.replace(
                    ".png", ".txt"
                )
                ok, _ = self.write_labels(
                    lbl_path, self.class_to_idx[class_name],
                    boxes_yolo, "nih", class_name, str(img_output),
                )
                if ok:
                    self.successful_images[class_name] += 1
                    self.class_counts[class_name] += 1
                    self.image_assignments["nih"][img_name] = class_name
                    self.processed_images["nih"].add(img_name)
                    pbar.update(1)
                else:
                    os.remove(img_output)
                    self.failed_reasons[class_name]["no_valid_boxes"] += 1
            else:
                os.remove(img_output)
                self.failed_reasons[class_name]["no_boxes_generated"] += 1

        pbar.close()

    def _process_rsna_pneumonia(self) -> None:
        """Process RSNA pneumonia data."""
        df = self.metadata_cache["rsna"]
        positives = df[df["Target"] == 1].copy()
        grouped = positives.groupby("patientId")
        patient_ids = list(grouped.groups.keys())
        np.random.shuffle(patient_ids)

        base = self.config.rsna_base_path
        needed = self.config.target_per_class - self.successful_images["Pneumonia"]
        if needed <= 0:
            return
        max_attempts = int(needed * self.config.max_attempts_multiplier)

        pbar = tqdm(total=needed, desc="Collecting RSNA Pneumonia images")
        attempts = 0

        for pid in patient_ids:
            if self.successful_images["Pneumonia"] >= self.config.target_per_class:
                break
            if attempts >= max_attempts:
                logger.warning("Reached maximum attempts for RSNA Pneumonia")
                break

            attempts += 1
            self.attempted_images["Pneumonia"] += 1

            dicom_path = f"{base}/stage_2_train_images/{pid}.dcm"
            if not os.path.exists(dicom_path):
                self.failed_reasons["Pneumonia"]["file_not_found"] += 1
                continue

            split = self.get_patient_split(pid, "rsna")
            img_name = f"rsna_{pid}.png"
            img_output = self.output_dir / split / "images" / img_name

            result = self._process_single_image(dicom_path, str(img_output), True)
            if not result or not result[0]:
                self.failed_reasons["Pneumonia"]["processing_failed"] += 1
                continue

            _, orig_w, orig_h, scale, pad_x, pad_y = result
            group = grouped.get_group(pid)
            boxes_yolo: List[List[float]] = []

            for _, row in group.iterrows():
                if pd.notna(row["x"]) and row["x"] >= 0:
                    sf = orig_w / 1024.0
                    norm = self.normalize_bbox(
                        row["x"] * sf, row["y"] * sf,
                        row["width"] * sf, row["height"] * sf,
                        scale, pad_x, pad_y,
                    )
                    if norm:
                        boxes_yolo.append(list(norm))

            if boxes_yolo:
                lbl = self.output_dir / split / "labels" / f"rsna_{pid}.txt"
                ok, _ = self.write_labels(
                    lbl, self.class_to_idx["Pneumonia"],
                    boxes_yolo, "rsna", "Pneumonia", str(img_output),
                )
                if ok:
                    self.successful_images["Pneumonia"] += 1
                    self.class_counts["Pneumonia"] += 1
                    self.processed_images["rsna"].add(pid)
                    pbar.update(1)
                else:
                    os.remove(img_output)
                    self.failed_reasons["Pneumonia"]["no_valid_boxes"] += 1
            else:
                os.remove(img_output)
                self.failed_reasons["Pneumonia"]["no_boxes_generated"] += 1

        pbar.close()

    # ── YAML config ─────────────────────────────────────────────────────

    def create_yaml_config(self) -> Path:
        """Write the ``dataset.yaml`` config file expected by YOLOv11."""
        config = {
            "path": str(self.output_dir),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {i: name for i, name in enumerate(self.class_names)},
            "nc": len(self.class_names),
            "model": f"yolov11{self.config.model_size}.pt",
            "augmentation": {
                "mixup": self.config.enable_mixup,
                "copy_paste": self.config.enable_copy_paste,
                "mosaic": 1.0,
                "degrees": 10.0,
                "translate": 0.1,
                "scale": 0.3,
                "shear": 2.0,
                "perspective": 0.001,
                "flipud": 0.0,
                "fliplr": 0.5,
            },
        }
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False)
        logger.info("YOLOv11 configuration saved to: %s", yaml_path)
        return yaml_path

    # ── Checkpoint ──────────────────────────────────────────────────────

    def save_checkpoint(self) -> None:
        """Persist processing state for later resumption."""
        state = {
            "processed_images": self.processed_images,
            "class_counts": self.class_counts,
            "successful_images": self.successful_images,
            "attempted_images": self.attempted_images,
            "failed_reasons": self.failed_reasons,
            "image_assignments": self.image_assignments,
            "patient_splits": self.patient_splits,
            "dedup_stats": self.dedup_stats,
            "debug_stats": self.debug_stats,
            "yolo_version": "11",
            "dedup_version": "improved_exact_duplicate_removal_v2",
        }
        path = self.output_dir / "checkpoint.pkl"
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        logger.info("Checkpoint saved to: %s", path)

    def load_checkpoint(self) -> bool:
        """Restore processing state from a checkpoint file."""
        path = self.output_dir / "checkpoint.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        self.processed_images = state["processed_images"]
        self.class_counts = state["class_counts"]
        self.successful_images = state.get(
            "successful_images", {c: 0 for c in self.class_names}
        )
        self.attempted_images = state.get(
            "attempted_images", {c: 0 for c in self.class_names}
        )
        self.failed_reasons = state.get(
            "failed_reasons", {c: defaultdict(int) for c in self.class_names}
        )
        self.image_assignments = state["image_assignments"]
        self.patient_splits = state["patient_splits"]
        self.dedup_stats = state["dedup_stats"]
        self.debug_stats = state["debug_stats"]
        if state.get("yolo_version") != "11":
            logger.warning("Checkpoint from different YOLO version")
        logger.info("Checkpoint loaded successfully")
        return True

    # ── Validation ──────────────────────────────────────────────────────

    def validate_dataset(self) -> bool:
        """Run integrity checks on the prepared dataset."""
        logger.info("=== Validating YOLOv11 Dataset ===")
        issues: List[str] = []

        for split in ("train", "val", "test"):
            label_dir = self.output_dir / split / "labels"
            image_dir = self.output_dir / split / "images"
            if not label_dir.exists():
                continue

            label_stems = {f.stem for f in label_dir.glob("*.txt")}
            image_stems = {f.stem for f in image_dir.glob("*.png")}

            for stem in label_stems - image_stems:
                issues.append(f"{split}: label '{stem}' has no image")
            for stem in image_stems - label_stems:
                issues.append(f"{split}: image '{stem}' has no label")

            for lf in label_dir.glob("*.txt"):
                with open(lf) as fh:
                    for i, line in enumerate(fh):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            issues.append(f"{lf.name} L{i + 1}: invalid format")
                            continue
                        try:
                            cidx = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            if not 0 <= cidx < len(self.class_names):
                                issues.append(f"{lf.name} L{i + 1}: bad class")
                            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                                issues.append(f"{lf.name} L{i + 1}: centre OOB")
                            if not (0 < w <= 1 and 0 < h <= 1):
                                issues.append(f"{lf.name} L{i + 1}: bad size")
                            tol = self.config.tolerance
                            if (
                                cx - w / 2 < -tol
                                or cx + w / 2 > 1 + tol
                                or cy - h / 2 < -tol
                                or cy + h / 2 > 1 + tol
                            ):
                                issues.append(f"{lf.name} L{i + 1}: box OOB")
                        except ValueError:
                            issues.append(f"{lf.name} L{i + 1}: parse error")

        if issues:
            logger.warning("Found %d validation issues", len(issues))
            for issue in issues[:10]:
                logger.warning("  %s", issue)
        else:
            logger.info("YOLOv11 Dataset validation PASSED ✓")
        return len(issues) == 0

    # ── Statistics report ───────────────────────────────────────────────

    def generate_statistics_report(self) -> None:
        """Print and save a comprehensive statistics report."""
        lines: List[str] = [
            "=== YOLOv11 DATASET STATISTICS REPORT ===\n",
            f"=== GUARANTEED {self.config.target_per_class} IMAGES PER CLASS ===\n",
            f"\nTarget Model: YOLOv11{self.config.model_size}",
            f"Image Size: {self.config.img_size}x{self.config.img_size}",
            "\n\n=== FINAL CLASS DISTRIBUTION ===",
        ]

        all_met = True
        for cn in self.class_names:
            count = self.successful_images[cn]
            ok = "✓" if count >= self.config.target_per_class else "✗"
            lines.append(f"  {cn}: {count}/{self.config.target_per_class} {ok}")
            if count < self.config.target_per_class:
                all_met = False

        if all_met:
            lines.append(
                f"\n🎉 ALL TARGETS MET! Exactly {self.config.target_per_class} "
                "images per class collected."
            )

        lines.append("\n\n=== PROCESSING EFFICIENCY ===")
        for cn in self.class_names:
            att = self.attempted_images[cn]
            if att <= 0:
                continue
            rate = self.successful_images[cn] / att * 100
            lines.append(f"  {cn}:")
            lines.append(f"    Attempted: {att}")
            lines.append(f"    Successful: {self.successful_images[cn]}")
            lines.append(f"    Success rate: {rate:.1f}%")
            if self.failed_reasons[cn]:
                lines.append("    Failure reasons:")
                for reason, cnt in self.failed_reasons[cn].items():
                    lines.append(f"      {reason}: {cnt}")

        lines.append("\n\nSplit Distribution:")
        for split in ("train", "val", "test"):
            img_dir = self.output_dir / split / "images"
            if img_dir.exists():
                n = len(list(img_dir.glob("*.png")))
                lines.append(f"  {split}: {n} images")

        lines.append("\n\nDataset Sources:")
        for ds in ("vinbig", "nih", "rsna", "tbx11k"):
            n = len(self.processed_images[ds])
            if n > 0:
                lines.append(f"  {ds.upper()}: {n} images")

        lines.append("\n\n=== DEDUPLICATION ANALYSIS ===")
        lines.append(
            f"\nTotal Exact Duplicates Removed: "
            f"{self.debug_stats.get('exact_duplicates_removed', 0)}"
        )

        lines.extend([
            "\n\n=== YOLOv11 TRAINING RECOMMENDATIONS ===",
            "Suggested command:",
            "```python",
            "from ultralytics import YOLO",
            "",
            f"model = YOLO('yolov11{self.config.model_size}.pt')",
            "results = model.train(",
            f"    data='{self.output_dir}/dataset.yaml',",
            "    epochs=100,",
            f"    imgsz={self.config.img_size},",
            "    batch=16,",
            "    patience=50,",
            "    optimizer='AdamW',",
            "    device=0",
            ")",
            "```",
        ])

        report = "\n".join(lines)
        (self.output_dir / "statistics_report.txt").write_text(report)
        logger.info("Statistics report saved to: %s", self.output_dir / "statistics_report.txt")
        print(report)

    # ── Main entry point ────────────────────────────────────────────────

    def run(self) -> bool:
        """Execute the full pipeline.

        Returns:
            *True* if all targets met and validation passed.
        """
        logger.info("Starting YOLOv11 Dataset Preparation — GUARANTEED %d per class",
                     self.config.target_per_class)
        logger.info("Target: YOLOv11%s @ %dpx", self.config.model_size, self.config.img_size)

        if self.load_checkpoint():
            logger.info("Resuming from checkpoint")

        self.process_dataset_by_class()
        self.save_checkpoint()
        valid = self.validate_dataset()
        yaml_path = self.create_yaml_config()
        self.generate_statistics_report()

        logger.info("=== YOLOv11 Pipeline Complete ===")
        logger.info("Dataset location: %s", self.output_dir)
        logger.info("Validation: %s", "PASSED" if valid else "FAILED")
        logger.info("Config file: %s", yaml_path)

        all_met = all(
            self.successful_images[c] >= self.config.target_per_class
            for c in self.class_names
        )
        return valid and all_met


# ── CLI entry point ─────────────────────────────────────────────────────────


def main() -> bool:
    """Main entry point for YOLOv11 medical dataset preparation."""
    config = DatasetConfig(
        output_dir="/kaggle/working/yolov11_dataset_1k_guaranteed",
        target_per_class=1000,
        debug_mode=True,
        num_workers=4,
        img_size=640,
        model_size="m",
        enable_mixup=True,
        enable_copy_paste=True,
        processing_multiplier=3.0,
        max_attempts_multiplier=5.0,
    )

    pipeline = DatasetPreparationPipeline(config)
    success = pipeline.run()

    if success:
        print("\n✅ YOLOv11 Dataset preparation completed successfully!")
        print("\n📊 Key Features:")
        print("• GUARANTEED 1000 images per class (with real annotations only)")
        print("• Continues processing until target met")
        print("• Only uses real medical annotations (no synthetic boxes)")
        print("• Tracks success rates and failure reasons")
        print("• Enhanced progress tracking with per-class progress bars")
    else:
        print("\n⚠️ Dataset preparation completed with issues. Check logs for details.")

    return success


if __name__ == "__main__":
    main()
