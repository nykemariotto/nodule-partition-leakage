"""
STAGE 2 — deterministic preprocessing (SPEC §6.2, §4).

For every labelled nodule (pylidc cluster, re-derived exactly as in metadata.py) and each
slice it spans, produce a 256x256x3 float32 array in [0,1]:
  1. HU volume via pylidc scan.to_volume() (rescale slope/intercept applied).
  2. Consensus ROI: pylidc.utils.consensus(clevel=preprocess.consensus_rule=0.5) => >=2/4
     annotator agreement mask + in-plane bbox; pad by preprocess.roi_pad_mm; square crop.
  3. HU clip (preprocess.hu_clip: [-1000, 400], air<->bone) -> rescale [0,1].
  4. Optional enhancement (preprocess.enhancement: none | clahe) — CONTROLLED ABLATION FACTOR.
  5. Resize to 256x256 (preprocess.input_size); replicate the single channel to 3.

Deterministic: no randomness anywhere (augmentation is TRAIN-only, STAGE 4). Held identical
across split arms. Writes .npy arrays + a processed index CSV.

    python -m src.preprocess --config config.yaml                 # full cohort
    python -m src.preprocess --config config.yaml --limit-scans 3 # validation subset
    python -m src.preprocess --config config.yaml --enhancement clahe
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pylidc as pl
from pylidc.utils import consensus
from skimage.transform import resize as sk_resize
from skimage.exposure import equalize_adapthist

from src.metadata import load_config, _consensus_label


# ------------------------------------------------------------------- image ops (pure)
def atomic_np_save(path, arr):
    """Crash-safe .npy write: temp + atomic rename. Never leaves a truncated file that a
    skip-if-exists check would mistake for a finished one."""
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_json_dump(path, obj):
    """Crash-safe JSON write: temp + atomic rename."""
    import json as _j
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        _j.dump(obj, f)
    os.replace(tmp, path)


def hu_clip(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip HU to [lo, hi] (air <-> bone) and rescale to [0,1]."""
    img = np.clip(img.astype(np.float32), lo, hi)
    return (img - lo) / (hi - lo)


def to_square(roi: np.ndarray) -> np.ndarray:
    """Center-pad a 2D ROI to a square with its minimum value (air), deterministic."""
    h, w = roi.shape
    s = max(h, w)
    out = np.full((s, s), roi.min(), dtype=roi.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = roi
    return out


def apply_enhancement(img01: np.ndarray, mode: str) -> np.ndarray:
    """Controlled ablation factor. 'none' = identity; 'clahe' = adaptive hist equalization.
    Input/output in [0,1]. Deterministic."""
    if mode == "none":
        return img01
    if mode == "clahe":
        return equalize_adapthist(img01, clip_limit=0.01).astype(np.float32)
    raise ValueError(f"unknown enhancement: {mode}")


def make_input(roi_hu: np.ndarray, cfg, enhancement: str) -> np.ndarray:
    """HU ROI -> (256,256,3) float32 in [0,1]: clip -> enhance -> square -> resize -> 3ch."""
    clip = cfg["preprocess"]["hu_clip"]
    size = cfg["preprocess"]["input_size"]
    win = hu_clip(roi_hu, clip[0], clip[1])
    win = apply_enhancement(win, enhancement)
    sq = to_square(win)
    rs = sk_resize(sq, (size[0], size[1]), order=1, mode="reflect",
                   anti_aliasing=True, preserve_range=True).astype(np.float32)
    rs = np.clip(rs, 0.0, 1.0)
    if cfg["preprocess"]["channels"] == "replicate":
        return np.repeat(rs[:, :, None], size[2], axis=2)
    raise ValueError("only channels: replicate is implemented")


# ------------------------------------------------------------------- per-nodule pipeline
def process_scan(scan, cfg, enhancement, out_root):
    """Re-cluster the scan (same order as metadata.py) and process every labelled nodule.
    Resumable: skips the scan if its per-scan rows JSON exists; writes it when done."""
    import json as _json
    rows_dir = os.path.join(out_root, "_rows")
    os.makedirs(rows_dir, exist_ok=True)
    # NOTE: keyed by patient AND series — 8 LIDC patients have 2 annotated series, and a
    # patient-only key silently skipped their second scan (data-loss bug, fixed 2026-07-18).
    rows_path = os.path.join(rows_dir, f"{scan.patient_id}_{scan.series_instance_uid[-8:]}.json")
    if os.path.exists(rows_path):
        return
    rows = []
    clx = cfg["metadata"]["clustering"]
    label_cfg = cfg["label"]
    principal = int(cfg["metadata"]["min_annotators_principal"])
    pad_mm = float(cfg["preprocess"]["roi_pad_mm"])
    clevel = float(cfg["preprocess"]["consensus_rule"])

    clusters = scan.cluster_annotations(metric=clx["metric"], tol=clx["tol"],
                                        factor=clx["factor"], min_tol=clx["min_tol"],
                                        verbose=False)
    if not clusters:
        atomic_json_dump(rows_path, rows)
        return
    series_short = scan.series_instance_uid[-8:]
    vol = None
    zvals = None
    pad_vox = int(round(pad_mm / float(scan.pixel_spacing)))

    for ni, cl in enumerate(clusters, start=1):
        scores = [int(a.malignancy) for a in cl if 1 <= int(a.malignancy) <= 5]
        label_name, encoded = _consensus_label(scores, label_cfg)
        if encoded is None:
            continue
        nodule_id = f"{scan.patient_id}_{series_short}_N{ni:02d}"
        try:
            cmask, cbbox, _ = consensus(cl, clevel=clevel, pad=None, ret_masks=True)
        except Exception as e:
            rows.append({"nodule_id": nodule_id, "status": f"consensus_error:{type(e).__name__}"})
            continue
        if vol is None:
            vol = scan.to_volume(verbose=False)              # HU volume (rows, cols, nslices)
            zvals = list(scan.slice_zvals)
        rows_i, cols_i, ks = cbbox
        i0 = max(rows_i.start - pad_vox, 0); i1 = min(rows_i.stop + pad_vox, vol.shape[0])
        j0 = max(cols_i.start - pad_vox, 0); j1 = min(cols_i.stop + pad_vox, vol.shape[1])
        nod_dir = os.path.join(out_root, nodule_id)
        os.makedirs(nod_dir, exist_ok=True)
        for kk in range(ks.start, ks.stop):
            if kk < 0 or kk >= vol.shape[2]:
                continue
            roi_hu = vol[i0:i1, j0:j1, kk]
            arr = make_input(roi_hu, cfg, enhancement)
            z = round(float(zvals[kk]), 2) if kk < len(zvals) else float(kk)
            fname = f"{nodule_id}_z{z:.2f}.npy"
            fpath = os.path.join(nod_dir, fname)
            atomic_np_save(fpath, arr)
            rows.append({
                "patient_id": scan.patient_id, "nodule_id": nodule_id,
                "z_position": z, "label": encoded, "label_name": label_name,
                "n_annotators": len(cl), "cohort_main": len(cl) >= principal,
                "enhancement": enhancement,
                "shape": "x".join(map(str, arr.shape)),
                "vmin": float(arr.min()), "vmax": float(arr.max()),
                "processed_path": str(Path(fpath).resolve()), "status": "ok",
            })
    atomic_json_dump(rows_path, rows)


def main():
    ap = argparse.ArgumentParser(description="STAGE 2 — deterministic preprocessing.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit-scans", type=int, default=0, help="0 = all; >0 = validation subset")
    ap.add_argument("--enhancement", default=None, help="override preprocess.enhancement_principal")
    args = ap.parse_args()

    cfg = load_config(args.config)
    enhancement = args.enhancement or cfg["preprocess"]["enhancement_principal"]
    if enhancement not in cfg["preprocess"]["enhancement"]:
        raise SystemExit(f"enhancement {enhancement} not in {cfg['preprocess']['enhancement']}")

    out_root = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"],
                            "processed", f"lidc_{enhancement}")
    os.makedirs(out_root, exist_ok=True)

    scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id, pl.Scan.id).all()
    scans = [s for s in scans if len(s.annotations) > 0]
    if args.limit_scans > 0:
        scans = scans[:args.limit_scans]

    print(f"[preprocess] enhancement={enhancement} · {len(scans)} annotated scans · out={out_root}")
    for i, s in enumerate(scans, 1):
        process_scan(s, cfg, enhancement, out_root)
        if i % 25 == 0 or i == len(scans):
            print(f"  {i}/{len(scans)} scans processed")

    # rebuild the index from ALL per-scan JSON row files (resumable + complete)
    import json as _json
    import glob as _glob
    rows = []
    for jf in _glob.glob(os.path.join(out_root, "_rows", "*.json")):
        with open(jf) as f:
            rows.extend(_json.load(f))
    idx = pd.DataFrame(rows)
    idx_path = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "metadata",
                            f"processed_index_{enhancement}.csv")
    idx.to_csv(idx_path, index=False)
    ok = idx[idx["status"] == "ok"] if len(idx) else idx
    print(f"\n[preprocess] wrote {len(ok)} processed slices; index -> {idx_path}")
    if len(ok):
        shapes = ok["shape"].value_counts().to_dict()
        print(f"  shapes: {shapes} · value range [{ok['vmin'].min():.3f}, {ok['vmax'].max():.3f}]")


if __name__ == "__main__":
    main()
