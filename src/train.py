"""
STAGE 4 — train ONE (dataset, arch, arm, seed) run (SPEC §6.4).

    python -m src.train --dataset lidc_binary --arch efficientnet_b0 --arm patient \
        --sample-unit slice --rep 0 --fold 0 --seed 42 --max-epochs 8

Saves (SPEC §0): y_true + y_prob (outputs/probs/*.npz), train/val history (outputs/history/*.json),
best weights (outputs/models/*.pt). Logs peak VRAM. 8 GB kit: AMP + grad-accum (effective 32) +
channels_last. Losing probabilities is what sank the previous submission.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score

from src.metadata import load_config
from src.models import build_model, input_size_for
from src.datasets import LidcDataset, load_processed_index
from src.config_contract import assert_contract, config_hash


def atomic_savez(path, **arrays):
    """Crash-safe .npz: temp + atomic rename (a truncated file must never look 'done')."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _load_split(cfg, tag, name):
    p = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "splits", f"{tag}_{name}.csv")
    return pd.read_csv(p)


@torch.no_grad()
def evaluate(model, loader, device, criterion, mem_fmt=torch.contiguous_format):
    model.eval()
    ys, ps, losses = [], [], []
    for x, y in loader:
        x = x.to(device, memory_format=mem_fmt, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast():
            out = model(x); loss = criterion(out, y)
        losses.append(loss.item())
        ps.append(torch.softmax(out.float(), 1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
    y_true = np.concatenate(ys); y_prob = np.concatenate(ps)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan")
    return float(np.mean(losses)), auc, y_true, y_prob


def main():
    ap = argparse.ArgumentParser(description="STAGE 4 — one training run.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default="lidc_binary")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--arm", default="patient", choices=["patient", "random", "nodule"])
    ap.add_argument("--sample-unit", default="slice", choices=["slice", "nodule"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enhancement", default=None)
    ap.add_argument("--max-epochs", type=int, default=0, help="0 = config train.max_epochs")
    ap.add_argument("--patience", type=int, default=0, help="0 = config; high value = effectively no early stop")
    ap.add_argument("--select", default="best_loss", choices=["best_loss", "best_auc", "final"],
                    help="checkpoint to evaluate: best val_loss (default) | best val_auc | final (most memorized)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # A config key no code reads is a false claim (D27). Block before touching the GPU: at grid
    # scale an undeclared key would otherwise be inherited silently by hundreds of runs.
    contract = assert_contract(cfg)
    cfg_hash = config_hash(cfg)                 # D25: stamp into every artifact for grid consistency
    print(f"[train] config-code contract OK: {contract} · config_sha256 {cfg_hash[:12]}")
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_cfg = cfg["train"]
    enh = args.enhancement or cfg["preprocess"]["enhancement_principal"]
    size = input_size_for(args.arch)
    tag = f"{args.dataset}_{args.sample_unit}_{args.arm}_rep{args.rep}_fold{args.fold}"

    pidx = load_processed_index(cfg, enh)
    ds_tr = LidcDataset(_load_split(cfg, tag, "train"), pidx, cfg, augment=True, size=size)
    ds_va = LidcDataset(_load_split(cfg, tag, "val"), pidx, cfg, augment=False, size=size)
    ds_te = LidcDataset(_load_split(cfg, tag, "test"), pidx, cfg, augment=False, size=size)
    nw = tr_cfg["num_workers"]
    pw = bool(tr_cfg.get("persistent_workers", False)) and nw > 0
    mem_fmt = torch.channels_last if tr_cfg.get("channels_last", False) else torch.contiguous_format
    dl_tr = DataLoader(ds_tr, batch_size=tr_cfg["batch_size"], shuffle=True, num_workers=nw,
                       pin_memory=True, drop_last=True, persistent_workers=pw)
    dl_va = DataLoader(ds_va, batch_size=tr_cfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=pw)
    dl_te = DataLoader(ds_te, batch_size=tr_cfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=pw)
    print(f"[train] {tag} · arch={args.arch} · enh={enh} · train {len(ds_tr)} / val {len(ds_va)} / test {len(ds_te)}")

    model = build_model(args.arch, num_classes=2, pretrained=True).to(device, memory_format=mem_fmt)
    # class-weighted loss (nodule-level can be imbalanced)
    ytr = np.array(ds_tr.labels)
    w = torch.tensor([1.0 / max((ytr == c).sum(), 1) for c in (0, 1)], dtype=torch.float32)
    w = (w / w.sum() * 2).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["lr"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tr_cfg["reduce_lr"]["factor"],
                                                       patience=tr_cfg["reduce_lr"]["patience"])
    scaler = GradScaler()
    accum = tr_cfg["grad_accum_steps"]
    max_epochs = args.max_epochs or tr_cfg["max_epochs"]
    # TRAINING budget guard. Defaults to the full budget (no truncation) because the `final`
    # memorisation probe (D22 ii) requires every run to reach max_epochs; truncating would leave
    # some runs without a probe and make the mechanism analysis asymmetric. The pre-registered
    # config.train.early_stopping_patience governs SELECTION (sel_patience below), not truncation.
    patience = args.patience or max_epochs

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history = []
    best_val, best_loss_state, bad = float("inf"), None, 0
    # CANONICAL SELECTION WINDOW (D46/D47). The pre-registered protocol is
    # config.train.early_stopping_patience (=10). The canonical checkpoint is the one THAT protocol
    # would select: argmin(val_loss) among epochs up to the point where patience-10 would have
    # stopped training. Training itself is NOT truncated -- it runs the full budget so the `final`
    # memorisation probe (D22 ii) exists for EVERY run and the mechanism analysis stays symmetric.
    # Selection is restricted; training is not.
    sel_patience = int(tr_cfg["early_stopping_patience"])
    sel_val, canon_state, canon_epoch, sel_bad, sel_frozen = float("inf"), None, None, 0, False
    for ep in range(1, max_epochs + 1):
        model.train(); opt.zero_grad(); tl = []
        for i, (x, y) in enumerate(dl_tr):
            x = x.to(device, memory_format=mem_fmt, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast():
                loss = criterion(model(x), y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tl.append(loss.item() * accum)
        vl, vauc, _, _ = evaluate(model, dl_va, device, criterion, mem_fmt)
        sched.step(vl)
        history.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_loss": vl,
                        "val_auc": vauc, "lr": opt.param_groups[0]["lr"]})
        print(f"  epoch {ep}: train_loss {np.mean(tl):.4f} · val_loss {vl:.4f} · val_auc {vauc:.4f}")
        improved_sel = (not sel_frozen) and vl < sel_val - 1e-4
        improved_best = vl < best_val - 1e-4
        snap = snapshot() if (improved_sel or improved_best) else None
        if not sel_frozen:
            if improved_sel:
                sel_val, canon_state, canon_epoch, sel_bad = vl, snap, ep, 0
            else:
                sel_bad += 1
                if sel_bad >= sel_patience:
                    sel_frozen = True
                    print(f"  [selection] patience-{sel_patience} window closed at epoch {ep}; "
                          f"canonical checkpoint fixed at epoch {canon_epoch}")
        if improved_best:
            best_val, best_loss_state, bad = vl, snap, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}"); break
    final_state = snapshot()
    if canon_state is None:                      # degenerate: val never improved inside the window
        canon_state, canon_epoch = best_loss_state, int(np.argmin([h["val_loss"] for h in history]) + 1)

    # --- DUAL CHECKPOINT (SPEC/DECISIONS): one training pass, two readings ---
    #   canonical = best val_loss WITHIN the patience-`selection_patience` window (D47) — what the
    #               pre-registered protocol would select; THE paper number. NOTE: this is not the
    #               unrestricted argmin; `unrestricted_best_epoch` is stamped alongside for audit.
    #   probe     = final          (max memorization; mechanistic only, never the headline)
    vls = [h["val_loss"] for h in history]
    unrestricted_best_epoch = int(np.argmin(vls) + 1)
    best_epoch = int(canon_epoch)     # the SELECTED canonical checkpoint (patience-window, D47)
    es_epoch, bv, bad_run = None, float("inf"), 0          # when patience WOULD have fired
    for h in history:
        if h["val_loss"] < bv - 1e-4:
            bv, bad_run = h["val_loss"], 0
        else:
            bad_run += 1
            if bad_run >= tr_cfg["early_stopping_patience"] and es_epoch is None:
                es_epoch = h["epoch"]
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    O = os.path.normpath(os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"]))
    for sub in ("probs", "history", "models", "metrics"):
        os.makedirs(os.path.join(O, sub), exist_ok=True)

    for sel_name, state in (("best_loss", canon_state), ("final", final_state)):
        if state is None:
            continue
        model.load_state_dict(state)
        te_loss, te_auc, y_true, y_prob = evaluate(model, dl_te, device, criterion, mem_fmt)
        acc = accuracy_score(y_true, (y_prob >= cfg["ensemble"]["threshold"]).astype(int))
        kind = "CANONICAL" if sel_name == "best_loss" else "PROBE"
        run = f"{tag}_{args.arch}_{enh}_seed{args.seed}" + ("" if sel_name == "best_loss" else f"_{sel_name}")
        print(f"[train] {kind:9} ({sel_name:9}) TEST auc {te_auc:.4f} · acc {acc:.4f} · loss {te_loss:.4f}")
        atomic_savez(os.path.join(O, "probs", f"{run}.npz"), y_true=y_true, y_prob=y_prob)
        atomic_json(os.path.join(O, "history", f"{run}.json"), {"tag": tag, "arch": args.arch, "checkpoint": sel_name, "kind": kind,
                       "config_sha256": cfg_hash,
                       "max_epochs": max_epochs, "patience": patience,
                       "training_patience": patience, "selection_patience": sel_patience,
                       "best_epoch": best_epoch,
                       "unrestricted_best_epoch": unrestricted_best_epoch,
                       "early_stop_would_fire_at": es_epoch,
                       "epochs_run": len(history), "history": history,
                       "test": {"auc": te_auc, "acc": acc, "loss": te_loss},
                       "peak_vram_gb": peak_vram})
        atomic_torch_save(os.path.join(O, "models", f"{run}.pt"), state)
    print(f"[train] saved BOTH checkpoints · best_epoch {best_epoch} · "
          f"early-stop would fire at {es_epoch} · peak VRAM {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()
