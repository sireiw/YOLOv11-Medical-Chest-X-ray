# YOLOv11 Medical Chest X-ray Dataset Preparation

A professional, modular pipeline for preparing chest X-ray datasets for **YOLOv11** object detection. It aggregates images from four public sources, applies intelligent bounding-box deduplication, and outputs a clean, balanced, YOLO-ready dataset.

## Detected Pathologies

| Class | Sources | Annotation Type |
|-------|---------|----------------|
| **Lung Cancer** (nodules/masses) | VinBigData, NIH ChestX-ray14 | Real bbox annotations |
| **Pneumonia** | RSNA Pneumonia Challenge, VinBigData | Real bbox annotations |
| **Tuberculosis** | TBX11K, VinBigData, NIH ChestX-ray14 | Real bbox annotations |

## Project Structure

```
yologithub/
├── src/
│   ├── __init__.py         # Package init
│   ├── config.py           # DatasetConfig dataclass & constants
│   ├── bbox_dedup.py       # Bounding-box deduplication (GIoU/DIoU/CIoU/Soft-NMS)
│   ├── pipeline.py         # DatasetPreparationPipeline + main()
│   └── visualization.py    # Notebook display utilities
├── yolo11new-only.ipynb    # Slim notebook: run pipeline + visualise
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### On Kaggle (recommended)

The pipeline is designed for the Kaggle notebook environment where datasets are mounted at `/kaggle/input/`. Simply run the notebook `yolo11new-only.ipynb`.

### Locally

Override dataset paths in the config:

```python
from src.config import DatasetConfig
from src.pipeline import DatasetPreparationPipeline

config = DatasetConfig(
    output_dir="./yolov11_dataset",
    nih_base_path="/path/to/nih",
    vinbig_base_path="/path/to/vinbig",
    rsna_base_path="/path/to/rsna",
    tbx11k_base_path="/path/to/tbx11k",
)
pipeline = DatasetPreparationPipeline(config)
pipeline.run()
```

## Datasets

Download and mount the following Kaggle datasets:

| Dataset | Link |
|---------|------|
| NIH ChestX-ray14 | [kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| VinBigData | [kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) |
| RSNA Pneumonia | [kaggle.com/c/rsna-pneumonia-detection-challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) |
| TBX11K | [kaggle.com/datasets/usmanshams/tbx11k-simplified](https://www.kaggle.com/datasets/usmanshams/tbx11k-simplified) |

## Key Features

- **Guaranteed target count** — processes extra images to meet the per-class target (default: 1000)
- **Patient-level splitting** — prevents data leakage between train/val/test
- **Advanced deduplication** — per-disease thresholds using GIoU, DIoU, CIoU, Soft-NMS, and clustering
- **DICOM support** — automatic windowing and conversion to PNG
- **Checkpoint/resume** — saves state for interrupted runs
- **Debug visualisation** — before/after dedup images saved automatically

## Training

After dataset preparation, train with [Ultralytics](https://docs.ultralytics.com/):

```python
from ultralytics import YOLO

model = YOLO("yolov11m.pt")
results = model.train(
    data="./yolov11_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50,
    optimizer="AdamW",
    device=0,
)
```

## Abstract

This project addresses the challenge of building a robust multi-class pathology detector for chest X-ray images. By aggregating annotations from four heterogeneous public datasets (VinBigData, NIH ChestX-ray14, RSNA, TBX11K), the pipeline produces a unified, balanced training set suitable for state-of-the-art YOLOv11 object detection. Special attention is given to annotation quality through a multi-stage bounding-box deduplication strategy that accounts for disease-specific lesion morphology.

**Keywords:** YOLOv11, chest X-ray, object detection, medical imaging, lung cancer, pneumonia, tuberculosis, dataset preparation, bounding-box deduplication
