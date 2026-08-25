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
    # ViT-S/16 replaced MaxViT-T as the second transformer before any run: it has direct
    # LIDC-IDRI precedent where MaxViT has none, and Swin (windowed, hierarchical, quasi-
    # convolutional) plus ViT (pure global attention) spans a far wider architectural range
    # than two conv/attention hybrids would.
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}


def build_model(name: str, num_classes: int = 2, pretrained: bool = True):
    if name not in _TIMM:
        raise ValueError(f"unknown architecture '{name}'; known: {sorted(_TIMM)}")
    return timm.create_model(_TIMM[name], pretrained=pretrained, num_classes=num_classes)


def input_size_for(name: str) -> int:
    """Input size fed to the model. 256 is our preprocessing size and the default.

    This list is maintained BY HAND on purpose, and must not be replaced by timm's own
    pretrained-config size. timm reports 224 for DenseNet-121 and EfficientNet-B0 as well, but the
    whole grid trained those at 256; deriving the size from timm would silently change the input of
    every convolutional run and invalidate all 210 of them. Only models that genuinely cannot take
    256 -- fixed-grid attention architectures -- are listed here.
    """
    return 224 if name in ("swin_tiny", "maxvit_tiny_tf_224", "vit_small") else 256
