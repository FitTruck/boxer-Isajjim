"""P1 sdp-ablation analysis over eval_ab variant reports.

Methodology (per the adversarial design review):
  - 35.png is byte-identical to 31.png -> excluded from all statistics.
  - Effective sample is IMAGES (n=5), not objects: per-image medians are the
    primary aggregation; pooled per-object sign counts are descriptive only.
  - A/A run (baseline vs baseline2) pins the MPS test-retest noise floor; any
    variant effect must clear it.
  - GT anchors come from scripts/eval_manifest.json (external standards only);
    axes with null GT are skipped. Reported per-anchor (n is tiny — a table,
    not a test).

Usage:
  python scripts/analyze_ablation.py baseline baseline2 nearest native14k native50k resized50k
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(_ROOT, "scripts", ".eval_cache", "reports")
MANIFEST = os.path.join(_ROOT, "scripts", "eval_manifest.json")
EXCLUDE_IMAGES = {"35"}  # byte-identical duplicate of 31


def _load(name: str) -> Dict[str, Any]:
    with open(os.path.join(REPORT_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _recs(rep: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in rep["records"] if r["image"] not in EXCLUDE_IMAGES]


def _gt_anchors() -> Dict[str, List[Dict[str, Any]]]:
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for im in manifest["images"]:
        stem = os.path.splitext(os.path.basename(im["path"]))[0]
        if im.get("objects"):
            out[stem] = im["objects"]
    return out


def _gt_errors(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Match anchors to highest-2D-score detection of the same label; per-axis
    |log(pred/gt)| with footprint long/short alignment; null axes skipped."""
    anchors = _gt_anchors()
    rows = []
    for stem, objs in anchors.items():
        img_recs = [r for r in recs if r["image"] == stem]
        for gt in objs:
            cands = sorted(
                (r for r in img_recs if r["label"].lower() == gt["label"].lower()),
                key=lambda r: -r["score2d"],
            )
            if not cands:
                continue
            r = cands[0]
            p_long, p_short = max(r["w_mm"], r["d_mm"]), min(r["w_mm"], r["d_mm"])
            gl, gs = max(gt["width_mm"], gt["depth_mm"]), min(gt["width_mm"], gt["depth_mm"])
            errs = {}
            if p_long > 0:
                errs["long"] = abs(math.log(p_long / gl))
            if p_short > 0:
                errs["short"] = abs(math.log(p_short / gs))
            if gt.get("height_mm") and r["h_mm"] > 0:
                errs["height"] = abs(math.log(r["h_mm"] / gt["height_mm"]))
            rows.append({
                "image": stem, "label": gt["label"], "prob": r["prob"],
                "pred": (round(r["w_mm"]), round(r["d_mm"]), round(r["h_mm"])),
                "gt": (gt["width_mm"], gt["depth_mm"], gt.get("height_mm")),
                "errs": {k: round(v, 3) for k, v in errs.items()},
                "mean_err": round(float(np.mean(list(errs.values()))), 3) if errs else None,
            })
    return rows


def main() -> int:
    names = sys.argv[1:] or [
        "baseline", "baseline2", "nearest", "native14k", "native50k", "resized50k"
    ]
    reps = {n: _load(n) for n in names}
    base = _recs(reps[names[0]])
    bmap = {(r["image"], r["det_index"]): r for r in base}

    # ---- A/A noise floor ---------------------------------------------------
    if "baseline2" in reps:
        aa = _recs(reps["baseline2"])
        diffs = []
        for r in aa:
            b = bmap.get((r["image"], r["det_index"]))
            if b:
                diffs.append(max(
                    abs(r["w_mm"] - b["w_mm"]), abs(r["d_mm"] - b["d_mm"]),
                    abs(r["h_mm"] - b["h_mm"]),
                ))
        print(f"=== A/A noise floor (baseline vs baseline2, n={len(diffs)} objs) ===")
        print(f"max |dim delta| = {max(diffs):.2f} mm, mean = {np.mean(diffs):.3f} mm\n")

    # ---- per-variant: paired deltas + per-image medians ---------------------
    print("=== paired vs baseline (objects pooled = descriptive; images = primary) ===")
    for name in names:
        if name in (names[0], "baseline2"):
            continue
        recs = _recs(reps[name])
        rows = []
        for r in recs:
            b = bmap.get((r["image"], r["det_index"]))
            if b is None:
                continue
            rows.append({
                "image": r["image"],
                "d_prior": r["prior_dev"] - b["prior_dev"],
                "d_depth": (abs(r["depth_log_ratio"]) - abs(b["depth_log_ratio"]))
                if r["depth_log_ratio"] is not None and b["depth_log_ratio"] is not None
                else None,
                "bounded": r["bounded"],
            })
        dp = [x["d_prior"] for x in rows if x["bounded"]]
        dd = [x["d_depth"] for x in rows if x["d_depth"] is not None]
        img_dp = {}
        for x in rows:
            if x["bounded"]:
                img_dp.setdefault(x["image"], []).append(x["d_prior"])
        img_medians = {k: float(np.median(v)) for k, v in img_dp.items()}
        n_imp = sum(1 for v in img_medians.values() if v < -1e-9)
        n_reg = sum(1 for v in img_medians.values() if v > 1e-9)
        print(f"\n[{name}] vs {names[0]}  (lift_s={sum(reps[name]['lift_seconds'].values()):.0f})")
        print(f"  prior_dev:  obj better/worse/tie = "
              f"{sum(1 for v in dp if v < -1e-9)}/{sum(1 for v in dp if v > 1e-9)}/"
              f"{sum(1 for v in dp if abs(v) <= 1e-9)}  mean_d={np.mean(dp):+.4f}")
        print(f"  |depth_lr|: obj better/worse = "
              f"{sum(1 for v in dd if v < -1e-9)}/{sum(1 for v in dd if v > 1e-9)}"
              f"  mean_d={np.mean(dd):+.4f}")
        print(f"  per-image prior_dev medians: "
              f"{ {k: round(v, 4) for k, v in sorted(img_medians.items())} }"
              f"  -> improved {n_imp}/{len(img_medians)} images, regressed {n_reg}")

    # ---- GT anchors ----------------------------------------------------------
    print("\n=== GT anchors (external standards; per-axis |log(pred/gt)|) ===")
    header = None
    table: Dict[str, Dict[str, Any]] = {}
    for name in names:
        if name == "baseline2":
            continue
        for row in _gt_errors(_recs(reps[name])):
            key = f"{row['image']}/{row['label']}"
            table.setdefault(key, {"gt": row["gt"]})[name] = (
                row["mean_err"], row["pred"], row["errs"]
            )
            header = True
    for key, by_var in table.items():
        print(f"\n{key}  gt={by_var.pop('gt')}")
        for name, (mean_err, pred, errs) in by_var.items():
            print(f"  {name:11s} mean|logerr|={mean_err}  pred={pred}  {errs}")
    if not header:
        print("(no anchors matched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
