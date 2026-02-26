"""YOLOv11 Medical Chest X-ray Detection Pipeline.

A professional, modular pipeline for training and deploying YOLOv11
on chest X-ray datasets (Tuberculosis, Pneumonia, Lung Cancer).

Modules:
    config              – DatasetConfig and constants
    bbox_dedup          – Bounding-box deduplication (IoU/GIoU/DIoU/CIoU)
    pipeline            – Dataset preparation pipeline
    visualization       – Dataset display utilities
    gpu_utils           – GPU memory management
    training            – YOLOv11 training with early stopping
    inference           – High-recall inference with TTA
    display_results     – Detection result display/annotation
    resume_training     – Resume/continue training from checkpoints
    training_viz        – Training loss visualisation
    diagnostics         – Label quality diagnostics
    metrics             – Recall/specificity analysis
    models              – UNet + EfficientNet neural network models
    three_phase_pipeline – 3-phase detection (YOLO → segmentation → classification)
"""

__version__ = "1.0.0"
