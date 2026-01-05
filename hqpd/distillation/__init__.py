"""Distillation utilities for SDXL UNet."""

from .distilled_unet import (
    create_distilled_unet,
    transfer_weights_from_teacher,
    DistilledUNetWrapper,
    FeatureExtractor,
    load_segmind_ssd1b,
    DISTILLED_UNET_CONFIG,
)

from .losses import (
    DistillationLoss,
    ProgressiveDistillationLoss,
    compute_distillation_step,
)

from .trainer import (
    DistillationTrainer,
    TrainingConfig,
    EMA,
    CosineAnnealingWithWarmup,
)

__all__ = [
    # Architecture
    "create_distilled_unet",
    "transfer_weights_from_teacher",
    "DistilledUNetWrapper",
    "FeatureExtractor",
    "load_segmind_ssd1b",
    "DISTILLED_UNET_CONFIG",
    # Losses
    "DistillationLoss",
    "ProgressiveDistillationLoss",
    "compute_distillation_step",
    # Training
    "DistillationTrainer",
    "TrainingConfig",
    "EMA",
    "CosineAnnealingWithWarmup",
]

