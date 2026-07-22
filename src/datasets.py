"""
STAGE 4 — torch Dataset reading from split CSVs (SPEC §3). NEVER flow_from_directory.

Reads a split CSV (sample rows: patient_id, nodule_id, z_position, label) and joins it to the
processed index to locate each 256x256x3 .npy. TRAIN-only augmentation (config.augment);
ImageNet normalization for the pretrained backbones.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_processed_index(cfg, enhancement):
    D = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "metadata")
    idx = pd.read_csv(os.path.join(D, f"processed_index_{enhancement}.csv"))
    idx = idx[idx["status"] == "ok"].copy()
    idx["z_position"] = idx["z_position"].round(2)
    return idx[["nodule_id", "z_position", "processed_path"]]


class LidcDataset(Dataset):
    def __init__(self, split_df, processed_index, cfg, augment=False, size=256):
        split_df = split_df.copy()
        split_df["z_position"] = split_df["z_position"].round(2)
        m = split_df.merge(processed_index, on=["nodule_id", "z_position"], how="inner")
        if len(m) < len(split_df):
            # HARD COVERAGE GATE. This was a printed WARNING before, and a grep in the training
            # driver hid it while 2.7% of slices were silently dropped from train AND test
            # (multi-series resumability bug). Training on incomplete data is now forbidden.
            missing = len(split_df) - len(m)
            gone = set(map(tuple, split_df[["nodule_id", "z_position"]].values)) - \
                   set(map(tuple, m[["nodule_id", "z_position"]].values))
            nods = sorted({n for n, _ in gone})[:5]
            raise RuntimeError(
                f"COVERAGE GATE FAILED: {missing}/{len(split_df)} split rows have no processed "
                f".npy ({missing/len(split_df)*100:.1f}%). Affected nodules (first 5): {nods}. "
                f"Re-run preprocessing until scripts/verify_coverage.py passes; do NOT train.")
        self.paths = m["processed_path"].tolist()
        self.labels = m["label"].astype(int).tolist()

        a = cfg["augment"]
        tfs = []
        if augment:
            tfs.append(T.RandomAffine(
                degrees=a["rotation_deg"],
                translate=(a["translate_frac"], a["translate_frac"]),
                scale=(1.0 - a["zoom"], 1.0 + a["zoom"]),
                shear=a["shear"] * 30.0))          # config shear (0.3) -> ~9 deg
            if a["hflip"]:
                tfs.append(T.RandomHorizontalFlip(0.5))
        if size != 256:
            tfs.append(T.Resize((size, size), antialias=True))
        tfs.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        self.tf = T.Compose(tfs)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = np.load(self.paths[i]).astype(np.float32)   # (256,256,3) in [0,1]
        x = torch.from_numpy(a).permute(2, 0, 1).contiguous()   # (3,256,256)
        return self.tf(x), self.labels[i]
