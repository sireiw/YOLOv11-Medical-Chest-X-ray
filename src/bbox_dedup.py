"""Advanced bounding-box deduplication for medical imaging.

Provides :class:`ImprovedBboxDeduplication` which combines multiple strategies
(GIoU, DIoU, CIoU, Soft-NMS, clustering) with disease-specific thresholds to
remove overlapping / duplicate annotations from chest X-ray datasets.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EPSILON, DatasetConfig

logger = logging.getLogger(__name__)


# ── Geometry helpers ────────────────────────────────────────────────────────

def to_corners(box: List[float]) -> Tuple[float, float, float, float]:
    """Convert YOLO centre-format ``[cx, cy, w, h]`` to corner-format.

    Returns:
        ``(x_min, y_min, x_max, y_max)``
    """
    cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _intersection_area(
    corners1: Tuple[float, float, float, float],
    corners2: Tuple[float, float, float, float],
) -> float:
    """Compute intersection area between two corner-format boxes."""
    inter_x_min = max(corners1[0], corners2[0])
    inter_y_min = max(corners1[1], corners2[1])
    inter_x_max = min(corners1[2], corners2[2])
    inter_y_max = min(corners1[3], corners2[3])

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)


# ── Main class ──────────────────────────────────────────────────────────────

class ImprovedBboxDeduplication:
    """Advanced bounding-box deduplication for medical imaging.

    Uses a per-disease pipeline:
      1. **Exact duplicate removal** — nearly identical centre & size.
      2. **Disease-specific strategy** — spatial protection + Soft-NMS for
         Lung Cancer, clustering or Soft-NMS for Pneumonia / TB.
      3. **Minimum-retention guard** — prevents over-aggressive merging.

    Args:
        config: Pipeline configuration (thresholds, weights, etc.).
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    # ── IoU family ──────────────────────────────────────────────────────

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Standard Intersection-over-Union."""
        c1, c2 = to_corners(box1), to_corners(box2)
        inter = _intersection_area(c1, c2)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - inter
        return inter / (union + EPSILON)

    def calculate_giou(self, box1: List[float], box2: List[float]) -> float:
        """Generalized IoU — penalises non-overlapping boxes more."""
        c1, c2 = to_corners(box1), to_corners(box2)
        iou = self.calculate_iou(box1, box2)

        # Smallest enclosing box
        enc_x_min = min(c1[0], c2[0])
        enc_y_min = min(c1[1], c2[1])
        enc_x_max = max(c1[2], c2[2])
        enc_y_max = max(c1[3], c2[3])
        enc_area = (enc_x_max - enc_x_min) * (enc_y_max - enc_y_min)

        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        inter = _intersection_area(c1, c2)
        union = area1 + area2 - inter

        return iou - (enc_area - union) / (enc_area + EPSILON)

    def calculate_diou(self, box1: List[float], box2: List[float]) -> float:
        """Distance-IoU — considers centre-point distance."""
        iou = self.calculate_iou(box1, box2)
        center_dist_sq = (box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2

        c1, c2 = to_corners(box1), to_corners(box2)
        enc_x_min = min(c1[0], c2[0])
        enc_y_min = min(c1[1], c2[1])
        enc_x_max = max(c1[2], c2[2])
        enc_y_max = max(c1[3], c2[3])
        diagonal_sq = (enc_x_max - enc_x_min) ** 2 + (enc_y_max - enc_y_min) ** 2 + EPSILON

        return iou - center_dist_sq / diagonal_sq

    def calculate_ciou(self, box1: List[float], box2: List[float]) -> float:
        """Complete IoU — adds aspect-ratio consistency term."""
        diou = self.calculate_diou(box1, box2)
        v = (4 / np.pi ** 2) * (
            np.arctan(box1[2] / (box1[3] + EPSILON))
            - np.arctan(box2[2] / (box2[3] + EPSILON))
        ) ** 2
        iou = self.calculate_iou(box1, box2)
        alpha = v / (1 - iou + v + EPSILON)
        return diou - alpha * v

    # ── Exact-duplicate removal ─────────────────────────────────────────

    def remove_exact_duplicates(
        self, boxes: List[List[float]], class_name: str
    ) -> List[List[float]]:
        """Remove boxes that are essentially at the same location."""
        if len(boxes) <= 1:
            return boxes

        thresholds = self.config.exact_duplicate_thresholds.get(
            class_name, {"location": 0.01, "size": 0.9}
        )
        loc_thr = thresholds["location"]
        size_thr = thresholds["size"]

        arr = np.array(boxes)
        n = len(arr)
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue

                dist = np.sqrt(
                    (arr[i, 0] - arr[j, 0]) ** 2 + (arr[i, 1] - arr[j, 1]) ** 2
                )
                if dist < loc_thr:
                    ratio_w = min(arr[i, 2], arr[j, 2]) / (max(arr[i, 2], arr[j, 2]) + EPSILON)
                    ratio_h = min(arr[i, 3], arr[j, 3]) / (max(arr[i, 3], arr[j, 3]) + EPSILON)

                    if ratio_w > size_thr and ratio_h > size_thr:
                        si = self.calculate_confidence_score(boxes[i], class_name)
                        sj = self.calculate_confidence_score(boxes[j], class_name)
                        if si >= sj:
                            keep[j] = False
                            logger.debug(
                                "Removing duplicate box at (%.3f, %.3f)",
                                arr[j, 0], arr[j, 1],
                            )
                        else:
                            keep[i] = False
                            logger.debug(
                                "Removing duplicate box at (%.3f, %.3f)",
                                arr[i, 0], arr[i, 1],
                            )
                            break

        return [boxes[i] for i in range(n) if keep[i]]

    # ── Shape / confidence scoring ──────────────────────────────────────

    def medical_shape_similarity(
        self, box1: List[float], box2: List[float], class_name: str
    ) -> float:
        """Shape similarity tuned for medical lesion characteristics."""
        area1, area2 = box1[2] * box1[3], box2[2] * box2[3]
        area_ratio = min(area1, area2) / (max(area1, area2) + EPSILON)

        ar1 = box1[2] / (box1[3] + EPSILON)
        ar2 = box2[2] / (box2[3] + EPSILON)
        ar_similarity = min(ar1, ar2) / (max(ar1, ar2) + EPSILON)

        weights = {
            "Lung_Cancer": (0.6, 0.4),
            "Pneumonia": (0.7, 0.3),
        }
        w_area, w_ar = weights.get(class_name, (0.65, 0.35))
        return w_area * area_ratio + w_ar * ar_similarity

    def calculate_confidence_score(
        self, box: List[float], class_name: str
    ) -> float:
        """Confidence score for medical lesion quality."""
        cx, cy, w, h = box
        weights = self.config.confidence_weights[class_name]

        # Location score
        if class_name == "Lung_Cancer":
            location_score = 0.8
        else:
            center_dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
            location_score = 1.0 - center_dist * 0.3

        # Size score
        area = w * h
        if class_name == "Lung_Cancer":
            if area < 0.005:
                size_score = 0.7
            elif area < 0.02:
                size_score = 0.9
            else:
                size_score = min(1.0, 0.9 + area)
        else:
            size_score = min(1.0, area * 10)

        # Aspect-ratio score
        aspect_ratio = max(w / h, h / w)
        max_ar = self.config.aspect_ratio_thresholds[class_name]
        if aspect_ratio <= 1.5:
            aspect_score = 1.0
        elif aspect_ratio <= max_ar:
            aspect_score = 1.0 - (aspect_ratio - 1.5) / (max_ar - 1.5) * 0.3
        else:
            aspect_score = 0.5

        return (
            weights["location"] * location_score
            + weights["size"] * size_score
            + weights["aspect"] * aspect_score
        )

    # ── Merge decision & helpers ────────────────────────────────────────

    def should_merge_boxes(
        self, box1: List[float], box2: List[float], class_name: str
    ) -> bool:
        """Whether two boxes should be merged (disease-specific)."""
        iou = self.calculate_iou(box1, box2)
        ciou = self.calculate_ciou(box1, box2)
        shape_sim = self.medical_shape_similarity(box1, box2, class_name)
        center_dist = np.sqrt(
            (box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2
        )

        if class_name == "Lung_Cancer":
            if center_dist < 0.003:
                rw = min(box1[2], box2[2]) / (max(box1[2], box2[2]) + EPSILON)
                rh = min(box1[3], box2[3]) / (max(box1[3], box2[3]) + EPSILON)
                if rw > 0.9 and rh > 0.9:
                    return True
            if self.is_box_contained(box1, box2, 0.98) or self.is_box_contained(
                box2, box1, 0.98
            ):
                return True
            if iou > 0.95 and shape_sim > 0.98:
                return True
            if center_dist > 0.02:
                return False
            return False

        elif class_name == "Pneumonia":
            if iou > 0.5 and shape_sim > 0.4:
                return True
            return ciou > 0.6

        else:  # Tuberculosis
            if iou > 0.55 and shape_sim > 0.5:
                return True
            return ciou > 0.65

    def is_box_contained(
        self,
        box1: List[float],
        box2: List[float],
        threshold: float = 0.9,
    ) -> bool:
        """Check if *box1* is mostly contained within *box2*."""
        c1, c2 = to_corners(box1), to_corners(box2)
        inter = _intersection_area(c1, c2)
        area1 = box1[2] * box1[3]
        return (inter / area1) > threshold

    def protect_spatially_separated_boxes(
        self, boxes: List[List[float]], min_separation: float
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Split boxes into *protected* (well-separated) and *remaining*."""
        if len(boxes) <= 1:
            return boxes, []

        protected: List[List[float]] = []
        remaining: List[List[float]] = []

        for i, box_i in enumerate(boxes):
            separated = all(
                np.sqrt(
                    (box_i[0] - boxes[j][0]) ** 2
                    + (box_i[1] - boxes[j][1]) ** 2
                )
                >= min_separation
                for j in range(len(boxes))
                if j != i
            )
            (protected if separated else remaining).append(box_i)

        return protected, remaining

    # ── Clustering & Soft-NMS ───────────────────────────────────────────

    def cluster_based_deduplication(
        self, boxes: List[List[float]], class_name: str
    ) -> List[List[float]]:
        """Cluster overlapping boxes and merge each cluster."""
        if len(boxes) <= 1:
            return boxes

        n = len(boxes)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                val = self.calculate_ciou(boxes[i], boxes[j])
                sim[i, j] = sim[j, i] = val

        clusters: List[List[int]] = []
        visited: set = set()
        threshold = self.config.iou_thresholds[class_name]

        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(n):
                if j not in visited and sim[i, j] > threshold:
                    if self.should_merge_boxes(boxes[i], boxes[j], class_name):
                        cluster.append(j)
                        visited.add(j)
            clusters.append(cluster)

        return [
            self.merge_box_cluster([boxes[idx] for idx in cl], class_name)
            if len(cl) > 1
            else boxes[cl[0]]
            for cl in clusters
        ]

    def merge_box_cluster(
        self, boxes: List[List[float]], class_name: str
    ) -> List[float]:
        """Merge a cluster into one representative box (weighted envelope)."""
        if len(boxes) == 1:
            return boxes[0]

        scores = [self.calculate_confidence_score(b, class_name) for b in boxes]
        # Weighted envelope
        corners = [to_corners(b) for b in boxes]
        margin = 0.95
        x_min = min(c[0] for c in corners)
        y_min = min(c[1] for c in corners)
        x_max = max(c[2] for c in corners)
        y_max = max(c[3] for c in corners)
        w = (x_max - x_min) * margin
        h = (y_max - y_min) * margin
        return [(x_min + x_max) / 2, (y_min + y_max) / 2, w, h]

    def soft_nms(
        self,
        boxes: List[List[float]],
        class_name: str,
        sigma: float = 0.5,
    ) -> List[List[float]]:
        """Soft-NMS with disease-aware sigma and minimum confidence."""
        if len(boxes) <= 1:
            return boxes

        if class_name == "Lung_Cancer":
            sigma = 0.1
            min_conf = 0.05
        else:
            min_conf = 0.3

        scored = sorted(
            [(b, self.calculate_confidence_score(b, class_name)) for b in boxes],
            key=lambda x: x[1],
            reverse=True,
        )

        keep: List[List[float]] = []
        while scored:
            cur_box, cur_score = scored.pop(0)
            if cur_score < min_conf:
                break
            keep.append(cur_box)

            remaining = []
            for other_box, other_score in scored:
                iou = self.calculate_iou(cur_box, other_box)
                if iou > 0:
                    new_score = other_score * np.exp(-(iou ** 2) / sigma)
                    if new_score >= min_conf:
                        remaining.append((other_box, new_score))
                else:
                    remaining.append((other_box, other_score))
            scored = sorted(remaining, key=lambda x: x[1], reverse=True)

        return keep

    # ── Main entry point ────────────────────────────────────────────────

    def deduplicate_boxes(
        self,
        boxes: List[List[float]],
        class_name: str,
        image_path: Optional[str] = None,
    ) -> Tuple[List[List[float]], Dict]:
        """Run the full deduplication pipeline on *boxes*.

        Returns:
            ``(deduplicated_boxes, stats_dict)``
        """
        if len(boxes) <= 1:
            return boxes, {
                "method": "none",
                "original": len(boxes),
                "final": len(boxes),
                "removed": 0,
                "removal_rate": 0.0,
                "exact_duplicates_removed": 0,
            }

        original_count = len(boxes)
        stats: Dict = {"original": original_count, "exact_duplicates_removed": 0}

        # Step 1 — exact duplicates
        boxes_after = self.remove_exact_duplicates(boxes, class_name)
        stats["exact_duplicates_removed"] = len(boxes) - len(boxes_after)
        if stats["exact_duplicates_removed"] > 0:
            logger.info(
                "Removed %d exact duplicates for %s",
                stats["exact_duplicates_removed"],
                class_name,
            )
        boxes = boxes_after

        if len(boxes) <= 1:
            stats.update(
                method="exact_duplicate_removal_only",
                final=len(boxes),
                removed=original_count - len(boxes),
                removal_rate=(
                    (original_count - len(boxes)) / original_count * 100
                    if original_count
                    else 0
                ),
            )
            return boxes, stats

        # Step 2 — disease-specific strategy
        if class_name == "Lung_Cancer":
            deduplicated, stats["method"] = self._dedup_lung_cancer(
                boxes, original_count, stats
            )
        elif class_name == "Pneumonia":
            if len(boxes) <= 5:
                deduplicated = self.soft_nms(boxes, class_name, sigma=0.5)
                stats["method"] = "exact_dedup_then_soft_nms"
            else:
                deduplicated = self.cluster_based_deduplication(boxes, class_name)
                stats["method"] = "exact_dedup_then_clustering"
        else:  # Tuberculosis
            deduplicated = self.cluster_based_deduplication(boxes, class_name)
            stats["method"] = "exact_dedup_then_clustering"

        # Step 3 — minimum-retention guard
        min_keep = {"Lung_Cancer": 0.75, "Pneumonia": 0.4, "Tuberculosis": 0.35}
        min_boxes = max(1, int(len(boxes) * min_keep.get(class_name, 0.3)))

        if len(deduplicated) < min_boxes and len(boxes) > min_boxes:
            if class_name == "Lung_Cancer":
                scored = sorted(
                    [(b, self.calculate_confidence_score(b, class_name)) for b in boxes],
                    key=lambda x: x[1],
                    reverse=True,
                )
                deduplicated = [b for b, _ in scored[:min_boxes]]
                stats["method"] = "exact_dedup_then_minimum_retention"
            else:
                deduplicated = self.soft_nms(boxes, class_name, sigma=0.8)[
                    :min_boxes
                ]
                stats["method"] = "exact_dedup_then_conservative_soft_nms"

        stats["final"] = len(deduplicated)
        stats["removed"] = original_count - len(deduplicated)
        stats["removal_rate"] = (
            (stats["removed"] / original_count * 100) if original_count else 0
        )
        return deduplicated, stats

    # ── Private helpers ─────────────────────────────────────────────────

    def _dedup_lung_cancer(
        self,
        boxes: List[List[float]],
        original_count: int,
        stats: Dict,
    ) -> Tuple[List[List[float]], str]:
        """Lung-cancer-specific dedup with spatial protection."""
        if len(boxes) <= self.config.lung_cancer_skip_dedup_threshold:
            return boxes, "exact_dedup_then_keep_all"

        protected, remaining = self.protect_spatially_separated_boxes(
            boxes, self.config.lung_cancer_min_separation
        )
        if remaining:
            deduped = protected + self.soft_nms(remaining, "Lung_Cancer", sigma=0.1)
            return deduped, "exact_dedup_then_spatial_soft_nms"
        return protected, "exact_dedup_then_spatial_protection"
