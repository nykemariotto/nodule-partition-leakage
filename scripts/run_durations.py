"""Wall-clock duration of every canonical training run, read from the local training logs.

Writes outputs/results/run_durations.csv and prints the ranges quoted in the paper. The training logs
(outputs/logs/) are not versioned, so the CSV is the released record: this script documents how it was
derived and cannot be re-run from a clone. The ten runs retrained after the patience check have no log line
for the retraining, so their duration is left blank rather than filled from the earlier training. Times
are wall-clock on one workstation, not a benchmark.

    python scripts/run_durations.py
"""
import csv
import glob
import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK_LINE = re.compile(r"^\W*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+OK\s+(lidc_binary\S+_seed42)\s+(\d+)s")
# Retrained after the patience check (see Quality control in the paper); the logs hold only the earlier training.
RETRAINED = {f"lidc_binary_{t}_none_seed42" for t in (
    "slice_random_rep0_fold4_efficientnet_b0", "slice_random_rep1_fold2_densenet121",
    "slice_random_rep1_fold3_densenet121", "slice_random_rep2_fold3_densenet121",
    "slice_random_rep1_fold2_efficientnet_b0", "nodule_random_rep2_fold2_densenet121",
    "nodule_patient_rep0_fold3_efficientnet_b0", "nodule_patient_rep1_fold0_efficientnet_b0",
    "nodule_patient_rep1_fold1_efficientnet_b0", "nodule_patient_rep2_fold3_efficientnet_b0")}
TAG = re.compile(r"lidc_binary_(ge3_)?(slice|nodule)_(patient|random|nodule)_rep(\d)_fold(\d)_(\w+?)_none_seed42$")


def latest_times():
    """{run tag: (timestamp, seconds)} from the latest OK line of each run across all logs."""
    latest = {}
    for path in glob.glob(os.path.join(ROOT, "outputs", "logs", "*.log")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = OK_LINE.match(line)
                if m and (m.group(2) not in latest or m.group(1) > latest[m.group(2)][0]):
                    latest[m.group(2)] = (m.group(1), int(m.group(3)))
    return latest


def main():
    times = latest_times()
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "outputs", "history", "*.json"))):
        if path.endswith("_final.json"):
            continue
        run = os.path.basename(path)[:-len(".json")]
        m = TAG.match(run)
        if not m:
            raise SystemExit(f"unexpected run name: {run}")
        with open(path, encoding="utf-8") as fh:
            hist = json.load(fh)
        rows.append({"run": run, "cohort": "ge3" if m.group(1) else "principal", "sample_unit": m.group(2),
                     "partition_unit": m.group(3), "repeat": m.group(4), "fold": m.group(5),
                     "architecture": m.group(6), "epochs_run": hist["epochs_run"],
                     "seconds": times[run][1] if run in times and run not in RETRAINED else ""})
    out = os.path.join(ROOT, "outputs", "results", "run_durations.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    timed = [r for r in rows if r["seconds"] != ""]
    print(f"{len(rows)} canonical runs, {len(timed)} with a logged duration -> {out}")
    print("without a logged duration:", [r["run"] for r in rows if r["seconds"] == ""])
    grid = [r for r in timed if r["cohort"] == "principal" and r["architecture"] != "swin_tiny"]
    for unit in ("nodule", "slice"):
        minutes = [r["seconds"] / 60 for r in grid if r["sample_unit"] == unit]
        print(f"principal cohort, {unit} sample unit: n={len(minutes)}, min {min(minutes):.1f}, "
              f"median {st.median(minutes):.1f}, max {max(minutes):.1f} minutes")


if __name__ == "__main__":
    main()
