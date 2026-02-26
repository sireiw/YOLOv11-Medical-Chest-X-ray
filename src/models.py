"""Neural network model definitions for the three-phase lung pipeline.

Contains:
  - :class:`UNetEfficientNet` — UNet with EfficientNet-B1 backbone for
    lung segmentation.
  - :class:`LungClassifier` — EfficientNet-B1-based classifier for lung
    disease classification.
  - :class:`SuppressOutput` — Context manager to silence stdout/stderr.
  - :func:`convert_numpy_types` — Recursively convert numpy types for
    JSON serialisation.
"""

import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None  # Optional dependency — only needed for segmentation


class UNetEfficientNet(nn.Module):
    """UNet model with EfficientNet-B1 backbone for lung segmentation.

    Requires ``segmentation-models-pytorch`` to be installed.

    Args:
        num_classes: Number of output segmentation classes.
        encoder_weights: Pre-trained encoder weights (e.g. ``'imagenet'``).
    """

    def __init__(
        self, num_classes: int = 1, encoder_weights: str = None
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation-models-pytorch is required: "
                "pip install segmentation-models-pytorch"
            )
        self.model = smp.Unet(
            encoder_name="efficientnet-b1",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LungClassifier(nn.Module):
    """EfficientNet-B1-based classifier for lung disease types.

    Replaces the default classifier head with a custom 2-layer MLP:
    ``Linear → ReLU → Dropout → Linear``.

    Args:
        num_classes: Number of output classes (default 4: Normal + 3 diseases).
        pretrained: Whether to use ImageNet-pretrained backbone.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = models.efficientnet_b1(pretrained=pretrained)
        num_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SuppressOutput:
    """Context manager to suppress stdout and stderr.

    Usage::

        with SuppressOutput():
            noisy_function()
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types.

    Useful for making dicts JSON-serialisable.

    Args:
        obj: Object to convert (dict, list, or scalar).

    Returns:
        Converted object with native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
