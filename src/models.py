"""
STAGE 4 — architecture factory (SPEC §2.4). ImageNet-initialised via timm, uniform API.

Config names map to timm model ids. Head is set to num_classes (2 for S2 binary).
All fit RTX 4060 Ti 8 GB at 256² with AMP + gradient accumulation.
"""
from __future__ import annotations

import timm

# config architecture name -> timm model id
_TIMM = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "convnext_tiny": "convnext_tiny",
    "inception_v3": "inception_v3",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "maxvit_tiny_tf_224": "maxvit_tiny_tf_224.in1k",
}


def build_model(name: str, num_classes: int = 2, pretrained: bool = True):
    if name not in _TIMM:
        raise ValueError(f"unknown architecture '{name}'; known: {sorted(_TIMM)}")
    return timm.create_model(_TIMM[name], pretrained=pretrained, num_classes=num_classes)


def input_size_for(name: str) -> int:
    """Most timm models here accept 256; swin/maxvit are trained at 224 (we resize inputs)."""
    return 224 if name in ("swin_tiny", "maxvit_tiny_tf_224") else 256
