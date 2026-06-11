"""Offline sanitizer-policy evaluation over a recorded eval_ab variant report.

The eval_ab `run` reports carry per-object RAW dims + BoxerNet prob (+ matched
GT when the manifest provides it), so sanitizer policies are compared without
re-running any model:

  raw     no correction
  clamp   legacy binary sanitize (in-range / clamp / severe->typical)
  fused   P2 prob-weighted continuous fusion + aspect-preserving scale branch

Also sweeps the 3D-confidence threshold policy for low-prob boxes:
  zero    prob < tau -> dims zeroed (current production semantics)
  prior   prob < tau -> class-typical midpoint dims (fused fallback)

Metrics on GT-matched objects:
  axis log-MAE   mean |log(pred/gt)| over footprint long/short + height
  vol MRE        mean |pred_volume - gt_volume| / gt_volume  (zeros count as 1.0)

Usage:
  python scripts/eval_sanitizer.py --report scripts/.eval_cache/reports/<name>.json
  python scripts/eval_sanitizer.py --report ... --table   # per-object review table
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402  (repo bootstrap: sys.path, BOXER_* env, logging)

from ai.pipeline.dimension_bounds import _BOUNDS, sanitize_dims  # noqa: E402

_TAUS = (0.0, 0.2, 0.3, 0.5)


def _typical_dims(label: str) -> Optional[Tuple[float, float, float]]:
    b = _BOUNDS.get(label.lower().strip())
    if not b:
        return None
    mid = lambda r: (r[0] + r[1]) / 2.0  # noqa: E731
    return mid(b["long"]), mid(b["short"]), mid(b["height"])


def _apply(policy: str, rec: Dict[str, Any]) -> Tuple[float, float, float]:
    w, d, h = rec["w_mm"], rec["d_mm"], rec["h_mm"]
    if policy == "raw":
        return w, d, h
    mode = "clamp" if policy == "clamp" else "fused"
    w2, d2, h2, _ = sanitize_dims(rec["label"], w, d, h, prob=rec["prob"], mode=mode)
    return w2, d2, h2


def _axis_errs(dims: Tuple[float, float, float], gt: Dict[str, float]) -> Optional[List[float]]:
    errs = ec.axis_log_errors(dims[0], dims[1], dims[2], gt)
    return list(errs.values()) if errs else None


def _vol_re(dims: Tuple[float, float, float], gt: Dict[str, float]) -> Optional[float]:
    if not gt.get("height_mm"):
        return None  # height not pinned by the standard -> no GT volume
    gv = gt["width_mm"] * gt["depth_mm"] * gt["height_mm"]
    pv = dims[0] * dims[1] * dims[2]
    return abs(pv - gv) / gv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--manifest", default=None, help="attach GT anchors from manifest")
    ap.add_argument("--table", action="store_true", help="print per-object review table")
    args = ap.parse_args()

    report = ec.load_report(args.report)
    records = ec.drop_dup(report["records"])
    if args.manifest:
        ec.attach_manifest_gt(records, args.manifest)
    gt_recs = [r for r in records if r.get("gt")]

    print(f"report={report['name']}  objects={len(records)}  gt-matched={len(gt_recs)}")

    # ---- policy comparison on GT-matched objects -------------------------
    if gt_recs:
        out: Dict[str, Any] = {}
        for policy in ("raw", "clamp", "fused"):
            axis_errs, vol_res = [], []
            for r in gt_recs:
                dims = _apply(policy, r)
                errs = _axis_errs(dims, r["gt"])
                if errs is not None:
                    axis_errs.append(float(np.mean(errs)))
                vr = _vol_re(dims, r["gt"])
                if vr is not None:
                    vol_res.append(vr)
            out[policy] = {
                "axis_logmae": round(float(np.mean(axis_errs)), 4) if axis_errs else None,
                "axis_logmae_median": round(float(np.median(axis_errs)), 4) if axis_errs else None,
                "vol_mre": round(float(np.mean(vol_res)), 4),
                "n": len(gt_recs),
            }
        print("\n=== sanitizer policies (GT objects) ===")
        print(json.dumps(out, indent=1))

        # paired fused vs clamp
        deltas = []
        for r in gt_recs:
            e_clamp = _axis_errs(_apply("clamp", r), r["gt"])
            e_fused = _axis_errs(_apply("fused", r), r["gt"])
            if e_clamp is not None and e_fused is not None:
                deltas.append(float(np.mean(e_fused)) - float(np.mean(e_clamp)))
        if deltas:
            better = sum(1 for x in deltas if x < -1e-9)
            worse = sum(1 for x in deltas if x > 1e-9)
            print(f"paired fused-vs-clamp: n={len(deltas)} better={better} "
                  f"worse={worse} mean_delta={np.mean(deltas):+.4f}")

        # ---- threshold sweep ---------------------------------------------
        print("\n=== low-prob threshold sweep (fused dims above tau) ===")
        sweep: Dict[str, Any] = {}
        for tau in _TAUS:
            for fallback in ("zero", "prior"):
                vol_res = []
                for r in gt_recs:
                    if r["prob"] < tau:
                        if fallback == "zero":
                            dims = (0.0, 0.0, 0.0)
                        else:
                            dims = _typical_dims(r["label"]) or (0.0, 0.0, 0.0)
                    else:
                        dims = _apply("fused", r)
                    vr = _vol_re(dims, r["gt"])
                    if vr is not None:
                        vol_res.append(vr)
                sweep[f"tau={tau}/{fallback}"] = {
                    "vol_mre": round(float(np.mean(vol_res)), 4),
                    "below_tau": sum(1 for r in gt_recs if r["prob"] < tau),
                }
        print(json.dumps(sweep, indent=1))
    else:
        print("(no GT-matched records — fill manifest objects and re-run eval_ab run)")

    # ---- per-object review table -----------------------------------------
    if args.table:
        print("\n=== per-object policies (all records) ===")
        fmt = "{:<10} {:<22} {:>5} | {:>18} | {:>18} | {:>18}"
        print(fmt.format("image", "label", "prob", "raw WxDxH", "clamp", "fused"))
        for r in sorted(records, key=lambda x: (x["image"], -x["score2d"])):
            row = []
            for policy in ("raw", "clamp", "fused"):
                w, d, h = _apply(policy, r)
                row.append(f"{w:.0f}x{d:.0f}x{h:.0f}")
            print(fmt.format(r["image"], r["label"][:22], f"{r['prob']:.2f}", *row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
