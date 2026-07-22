"""
CONFIG-CODE CONTRACT — a config key that no code reads is a false claim, not documentation.

DECISIONS D27 / D23. The previous submission was rejected because the manuscript described
methodology the code did not implement. `config.yaml` is the manuscript's machine-readable face,
so every key in it must have a DECLARED, auditable relationship to the code. On 2026-07-20 an
audit found 44 of 91 keys read by nothing at all — including `split.stratify_key: label`
(no stratification existed) and `train.freeze_then_finetune: true` (no freezing code existed).
Both were removed (D27, D28).

This module makes that class of defect impossible to reintroduce silently — which matters most
at grid scale, where hundreds of runs would otherwise inherit a false claim.

STATUS VOCABULARY — every key carries exactly one:
  consumed       code reads the key. Verified STATICALLY by scripts/verify_config_contract.py.
  hardcoded      code implements the declared value but does not read the key. Behaviour matches,
                 the config is NOT the source of truth. Tech debt: safe today, a lie the moment
                 someone edits the config expecting an effect.
  documentation  describes the study; no runtime behaviour is claimed.
  deferred       belongs to a stage that is NOT implemented. MUST NOT be described as done in
                 the manuscript.

`assert_contract` blocks on any drift in either direction: a config key with no declaration, or
a declaration for a key that no longer exists. Per D23 it raises — it does not warn.
"""
from __future__ import annotations


CONTRACT: dict[str, tuple[str, str]] = {
    'project.name': ('documentation', 'study name; no runtime behaviour'),
    'project.root': ("consumed", ""),
    'paths.lidc_dicom_root': ("consumed", ""),
    'paths.lidc_metadata_csv': ("consumed", ""),
    'paths.lndb_root': ('deferred', 'LNDb (setting S3) not implemented'),
    'paths.outputs': ("consumed", ""),
    'metadata.engine': ("consumed", ""),
    'metadata.emit_superset': ('hardcoded', 'src/metadata.py always emits the superset master table'),
    'metadata.min_annotators_principal': ("consumed", ""),
    'metadata.min_annotators_sensitivity': ("consumed", ""),
    'metadata.over_merge_flag_threshold': ("consumed", ""),
    'metadata.clustering.metric': ("consumed", ""),
    'metadata.clustering.tol': ("consumed", ""),
    'metadata.clustering.factor': ("consumed", ""),
    'metadata.clustering.min_tol': ("consumed", ""),
    'metadata.io_workers': ("consumed", ""),
    'label.malignant': ("consumed", ""),
    'label.benign': ("consumed", ""),
    'label.exclude_ambiguous': ('hardcoded', 'median binarisation in src/metadata.py always drops score 3 (D1)'),
    'label.consensus': ("consumed", ""),
    'label.encoding.malignant': ("consumed", ""),
    'label.encoding.benign': ("consumed", ""),
    'experiment.settings.S1.dataset': ('documentation', 'describes setting S1'),
    'experiment.settings.S1.task': ('documentation', 'describes setting S1'),
    'experiment.settings.S1.sample_units': ('documentation', 'describes setting S1'),
    'experiment.settings.S2.dataset': ('documentation', 'describes setting S2'),
    'experiment.settings.S2.task': ('documentation', 'describes setting S2'),
    'experiment.settings.S2.sample_units': ('documentation', 'describes setting S2'),
    'experiment.settings.S3.dataset': ('documentation', 'describes setting S3'),
    'experiment.settings.S3.task': ('documentation', 'describes setting S3'),
    'experiment.settings.S3.sample_units': ('documentation', 'describes setting S3'),
    'experiment.split_units': ('documentation', 'describes the 2x2; the arm is a CLI argument'),
    'experiment.report_leakage_levels_separately': ('documentation', 'reporting rule (D8), enforced by hand not by code'),
    'preprocess.input_size': ("consumed", ""),
    'preprocess.hu_clip': ("consumed", ""),
    'preprocess.rescale': ('hardcoded', 'src/preprocess.py always emits 256x256x3 in [0,1]'),
    'preprocess.consensus_rule': ("consumed", ""),
    'preprocess.roi_pad_mm': ("consumed", ""),
    'preprocess.channels': ("consumed", ""),
    'preprocess.enhancement': ("consumed", ""),
    'preprocess.enhancement_principal': ("consumed", ""),
    'split.ratios.train': ("consumed", ""),
    'split.ratios.val': ("consumed", ""),
    'split.ratios.test': ("consumed", ""),
    'split.group_key': ('hardcoded', 'src/splits.py hardcodes patient_id; same value, but config is not the source'),
    'repetition.scheme': ('documentation', 'names the design (D5); realised by --rep/--fold arguments'),
    'repetition.variance_correction': ('hardcoded', 'src/stats.py implements Nadeau-Bengio; rho passed as an argument'),
    'repetition.per_setting.S2.repeats': ('deferred', 'grid sizing for S2; the subset stage uses CLI arguments'),
    'repetition.per_setting.S2.folds': ('deferred', 'grid sizing for S2; the subset stage uses CLI arguments'),
    'repetition.per_setting.S1.repeats': ('deferred', 'grid sizing for S1; the subset stage uses CLI arguments'),
    'repetition.per_setting.S1.folds': ('deferred', 'grid sizing for S1; the subset stage uses CLI arguments'),
    'repetition.per_setting.S3.repeats': ('deferred', 'grid sizing for S3; the subset stage uses CLI arguments'),
    'repetition.per_setting.S3.folds': ('deferred', 'grid sizing for S3; the subset stage uses CLI arguments'),
    'repetition.seed_list': ("consumed", ""),
    'repetition.group_key': ('hardcoded', 'duplicate of split.group_key; same hardcoding'),
    'seed_list': ("consumed", ""),
    'train.optimizer': ('hardcoded', 'src/train.py hardcodes Adam'),
    'train.lr': ("consumed", ""),
    'train.batch_size': ("consumed", ""),
    'train.grad_accum_steps': ("consumed", ""),
    'train.max_epochs': ("consumed", ""),
    'train.early_stopping_patience': ("consumed", ""),
    'train.checkpoint_selection': ("consumed", ""),
    'train.reduce_lr.factor': ("consumed", ""),
    'train.reduce_lr.patience': ("consumed", ""),
    'train.reduce_lr.monitor': ('hardcoded', 'src/train.py always steps the scheduler on val_loss'),
    'train.amp': ('hardcoded', 'src/train.py always uses autocast; the key cannot turn it off'),
    'train.channels_last': ("consumed", ""),
    'train.num_workers': ("consumed", ""),
    'train.persistent_workers': ("consumed", ""),
    'augment.rotation_deg': ("consumed", ""),
    'augment.translate_frac': ("consumed", ""),
    'augment.shear': ("consumed", ""),
    'augment.zoom': ("consumed", ""),
    'augment.hflip': ("consumed", ""),
    'architectures.primary': ('deferred', 'only densenet121 + efficientnet_b0 are run; arch is a CLI argument'),
    'architectures.sota': ('deferred', 'transformer baselines not run yet'),
    'ensemble.members': ('deferred', 'ensembling stage not implemented'),
    'ensemble.method': ('deferred', 'ensembling stage not implemented'),
    'ensemble.threshold': ("consumed", ""),
    'ensemble.sensitivity': ('deferred', 'ensembling stage not implemented'),
    'evaluate.aggregation': ('deferred', 'slice + nodule implemented; PATIENT level never computed (D7)'),
    'evaluate.bootstrap_n': ("consumed", ""),
    'evaluate.ci': ("consumed", ""),
    'evaluate.primary_metric': ('documentation', 'AUC is primary by D21/D26; not read as a switch'),
    'stats.paired_ab': ('hardcoded', 'src/stats.py implements Wilcoxon (D5)'),
    'stats.within_arm': ('deferred', 'McNemar not implemented; no within-arm comparison reported yet'),
    'mechanism.attribution': ('deferred', 'mechanism analysis (SPEC 2.6) not implemented'),
    'mechanism.quantify_leakage': ('deferred', 'mechanism analysis (SPEC 2.6) not implemented'),
}

VALID_STATUS = {"consumed", "hardcoded", "documentation", "deferred"}


def _walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    else:
        yield path


def config_keys(cfg) -> list[str]:
    """Every leaf key path in the config, dotted."""
    return list(_walk(cfg))


def assert_contract(cfg) -> dict:
    """Raise unless every config key is declared and every declaration still exists.

    Called at the start of every run (src/train.py). Returns a status->count summary so a caller
    can log what the run is operating under.
    """
    keys = set(config_keys(cfg))
    declared = set(CONTRACT)

    undeclared = sorted(keys - declared)
    orphaned = sorted(declared - keys)
    bad = sorted(k for k, (s, _) in CONTRACT.items() if s not in VALID_STATUS)

    problems = []
    if undeclared:
        problems.append(
            f"{len(undeclared)} config key(s) with NO declaration in the contract: {undeclared}. "
            f"Add each to CONTRACT with a status, or delete it from config.yaml. A key no code "
            f"reads is a false claim (D27).")
    if orphaned:
        problems.append(
            f"{len(orphaned)} contract entr(ies) for key(s) no longer in config.yaml: {orphaned}. "
            f"Remove them so the contract cannot drift from the config.")
    if bad:
        problems.append(f"invalid status for {bad}; allowed: {sorted(VALID_STATUS)}")
    if problems:
        raise RuntimeError("CONFIG-CODE CONTRACT FAILED. " + " | ".join(problems))

    summary: dict[str, int] = {}
    for _, (status, _reason) in CONTRACT.items():
        summary[status] = summary.get(status, 0) + 1
    return summary


def deferred_keys() -> list[str]:
    """Keys whose stage is NOT implemented — never describe these as done in the manuscript."""
    return sorted(k for k, (s, _) in CONTRACT.items() if s == "deferred")


def config_hash(cfg) -> str:
    """SHA-256 of the PARSED config values (D25 config-staleness / grid consistency).

    Hashes the canonicalised VALUES (sorted-key JSON), not the file bytes, so a comment or
    whitespace edit does not trip it but any change to a key or value does. Stamped into every
    run's history by src/train.py and checked by scripts/run_is_valid.py so that no run trained
    under a different config is silently folded into the grid average. The whole grid must share
    one hash; a mid-grid config change makes subsequent skip-decisions reject until re-confirmed.
    """
    import hashlib
    import json
    payload = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
