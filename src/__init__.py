"""YOLOv11 Medical Imaging Dataset Preparation Pipeline.

Prepares chest X-ray datasets (VinBigData, NIH, RSNA, TBX11K) for object
detection with YOLOv11. Supports three pathology classes:
  - Lung Cancer (nodules/masses)
  - Pneumonia
  - Tuberculosis

Features:
  - Advanced bounding-box deduplication (GIoU, DIoU, CIoU, Soft-NMS)
  - Patient-level train/val/test splits to prevent data leakage
  - DICOM-to-PNG conversion with proper windowing
  - Checkpoint/resume support
  - Debug visualizations for deduplication analysis
"""

__version__ = "1.0.0"
