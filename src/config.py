"""Configuration module for YOLOv11 dataset preparation.

Defines ``DatasetConfig``, a dataclass holding all tuneable parameters for
image processing, bounding-box deduplication, dataset paths, and YOLOv11
training recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON: float = 1e-6
"""Small constant used throughout the pipeline to avoid division by zero."""

CLASS_NAMES: List[str] = ["Lung_Cancer", "Pneumonia", "Tuberculosis"]
"""Ordered list of detection class names (index == YOLO class id)."""

# Default dataset base paths (Kaggle layout)
DEFAULT_NIH_BASE: str = "/kaggle/input/data"
DEFAULT_VINBIG_BASE: str = (
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
)
DEFAULT_RSNA_BASE: str = "/kaggle/input/rsna-pneumonia-detection-challenge"
DEFAULT_TBX11K_BASE: str = "/kaggle/input/tbx11k-simplified/tbx11k-simplified"


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Enhanced configuration for YOLOv11 dataset preparation.

    All paths default to the standard Kaggle competition layout but can be
    overridden for local development.

    Attributes:
        output_dir: Root directory for the prepared dataset.
        img_size: Target image size (pixels, square).
        target_per_class: Number of images to collect per class.
        age_range: (min, max) patient age filter for NIH.
        split_ratios: Train / Val / Test ratio.
        debug_mode: If True, enable verbose logging and visualizations.
        tolerance: Floating-point comparison tolerance for validation.
        num_workers: Parallel workers for image processing.
        processing_multiplier: Process N× target to account for failures.
        max_attempts_multiplier: Give up after N× target attempts.
    """

    # --- General ---------------------------------------------------------
    output_dir: str = "/kaggle/working/yolov11_dataset_1k_fixed"
    img_size: int = 640
    target_per_class: int = 1000
    age_range: Tuple[int, int] = (18, 90)
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    debug_mode: bool = True
    tolerance: float = 1e-6
    num_workers: int = 4

    # --- Processing multipliers ------------------------------------------
    processing_multiplier: float = 3.0
    max_attempts_multiplier: float = 5.0

    # --- Dataset paths (overridable) -------------------------------------
    nih_base_path: str = DEFAULT_NIH_BASE
    vinbig_base_path: str = DEFAULT_VINBIG_BASE
    rsna_base_path: str = DEFAULT_RSNA_BASE
    tbx11k_base_path: str = DEFAULT_TBX11K_BASE

    # --- Medical-specific deduplication thresholds -----------------------
    iou_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "Lung_Cancer": 0.05,
            "Pneumonia": 0.45,
            "Tuberculosis": 0.40,
        }
    )

    min_center_distances: Dict[str, float] = field(
        default_factory=lambda: {
            "Lung_Cancer": 0.008,
            "Pneumonia": 0.05,
            "Tuberculosis": 0.04,
        }
    )

    size_similarity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "Lung_Cancer": 0.85,
            "Pneumonia": 0.3,
            "Tuberculosis": 0.35,
        }
    )

    aspect_ratio_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "Lung_Cancer": 2.0,
            "Pneumonia": 3.0,
            "Tuberculosis": 2.5,
        }
    )

    confidence_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "Lung_Cancer": {"location": 0.1, "size": 0.4, "aspect": 0.5},
            "Pneumonia": {"location": 0.3, "size": 0.5, "aspect": 0.2},
            "Tuberculosis": {"location": 0.25, "size": 0.5, "aspect": 0.25},
        }
    )

    lung_cancer_skip_dedup_threshold: int = 5
    lung_cancer_min_separation: float = 0.1

    exact_duplicate_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "Lung_Cancer": {"location": 0.003, "size": 0.95},
            "Pneumonia": {"location": 0.01, "size": 0.9},
            "Tuberculosis": {"location": 0.008, "size": 0.92},
        }
    )

    # --- YOLOv11-specific ------------------------------------------------
    model_size: str = "m"
    enable_mixup: bool = True
    enable_copy_paste: bool = True
