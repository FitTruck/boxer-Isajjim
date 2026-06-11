"""A/B evaluation harness with frozen detection + depth inputs.

Motivation: BoxerNet-stage changes (sdp sampling, thresholds, ...) must be
measured without detector/depth run-to-run variance. This harness freezes the
OWLv2 detections and Depth Pro depth map per image once (snapshot), then runs
BoxerNet variants against the identical cached inputs — a strictly paired
design where per-object deltas are attributable to the variant alone.

Stages:
  snapshot  Run OWLv2 + Depth Pro once per manifest image; cache boxes/scores/
            labels/focal (JSON) and the metric depth map (.npy) to disk.
  run       Load BoxerLifter with variant sdp params (conf_threshold=0 so every
            input box is recorded; thresholds are swept post-hoc). Record raw
            dims + prob per object. The sanitizer is NOT applied; the would-be
            corrections are computed as metrics instead.
  compare   Join variants per (image, det_index), report aggregate metrics and
            paired deltas vs the baseline with an exact two-sided sign test.

Metrics (no real GT required):
  prior_dev        mean |log(raw/sanitized)| over the 3 axes — magnitude of
                   disagreement with the class physical bounds (0 = in range).
                   Bounds are not used at inference, so for sdp variants this
                   is a non-circular proxy.
  depth_log_ratio  log(predicted OBB center depth / observed median bbox
                   depth). Biased positive by half the object extent, but the
                   bias cancels in paired comparison.
  gt_err           |log(pred/gt)| per axis (footprint long/short aligned;
                   height direct) when the manifest carries GT objects.

Usage:
  python scripts/eval_ab.py snapshot --manifest scripts/eval_manifest.json
  python scripts/eval_ab.py run --name baseline
  python scripts/eval_ab.py compare before after
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402  (repo bootstrap: sys.path, BOXER_* env, logging)

log = logging.getLogger("eval_ab")

SNAP_DIR = ec.SNAP_DIR
REPORT_DIR = ec.REPORT_DIR
DEFAULT_MANIFEST = ec.DEFAULT_MANIFEST


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _filter_only(images: List[Dict[str, Any]], only: Optional[str]) -> List[Dict[str, Any]]:
    if not only:
        return images
    keep = {s.strip() for s in only.split(",") if s.strip()}
    return [im for im in images if ec.stem(im["path"]) in keep]


# --------------------------------------------------------------------------- #
# Stage 1: snapshot (OWLv2 + Depth Pro, cached once)
# --------------------------------------------------------------------------- #
def cmd_snapshot(args: argparse.Namespace) -> int:
    from PIL import Image

    from ai.pipeline import DepthEstimator, Owlv2Detector

    images = _filter_only(ec.load_manifest(args.manifest), args.only)
    os.makedirs(SNAP_DIR, exist_ok=True)

    detector = Owlv2Detector()
    depth_model = DepthEstimator()

    for item in images:
        path, stem = item["path"], ec.stem(item["path"])
        img = Image.open(path).convert("RGB")

        t0 = time.perf_counter()
        det = detector.detect(img)
        det_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        dres = depth_model.estimate(img)
        depth_s = time.perf_counter() - t0

        np.save(os.path.join(SNAP_DIR, f"{stem}_depth.npy"), dres.depth.astype(np.float32))
        meta = {
            "path": path,
            "split": item.get("split", "dev"),
            "width": img.width,
            "height": img.height,
            "focal_px": dres.focal_length_px,
            "boxes": [[float(v) for v in b] for b in det["boxes"]],
            "scores": [float(s) for s in det["scores"]],
            "labels": list(det["labels"]),
            "det_s": round(det_s, 2),
            "depth_s": round(depth_s, 2),
        }
        with open(os.path.join(SNAP_DIR, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        log.info(
            "[snapshot] %s: %d boxes (%.1fs det, %.1fs depth, focal=%s)",
            stem, len(meta["boxes"]), det_s, depth_s, dres.focal_length_px,
        )
    return 0


# --------------------------------------------------------------------------- #
# Stage 2: run one BoxerNet variant against the cached inputs
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    from PIL import Image

    from ai.pipeline import BoxerLifter
    from ai.pipeline.dimension_bounds import _BOUNDS, sanitize_dims

    images = _filter_only(ec.load_manifest(args.manifest), args.only)
    os.makedirs(REPORT_DIR, exist_ok=True)

    lifter = BoxerLifter(conf_threshold=0.0)
    if lifter.net is None:
        log.error("BoxerNet unavailable (checkpoint/repo missing) — abort")
        return 2
    # Camera->world default rotation used to read back the predicted center
    # depth in the camera frame (harness runs the level-camera default path).
    R = BoxerLifter._R_WORLD_FROM_CAM

    all_records: List[Dict[str, Any]] = []
    times: Dict[str, float] = {}
    for item in images:
        stem = ec.stem(item["path"])
        snap_path = os.path.join(SNAP_DIR, f"{stem}.json")
        if not os.path.exists(snap_path):
            log.warning("no snapshot for %s — run `snapshot` first; skipping", stem)
            continue
        with open(snap_path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        depth = np.load(os.path.join(SNAP_DIR, f"{stem}_depth.npy"))
        img = Image.open(snap["path"]).convert("RGB")
        boxes = np.asarray(snap["boxes"], dtype=np.float32)
        if len(boxes) == 0:
            continue

        t0 = time.perf_counter()
        obbs = lifter.lift(
            image=img,
            bboxes_xyxy=boxes,
            labels=snap["labels"],
            depth=depth,
            focal_length_px=snap["focal_px"],
        )
        times[stem] = time.perf_counter() - t0

        by_idx = {o.input_index: o for o in obbs}
        records: List[Dict[str, Any]] = []
        for i, label in enumerate(snap["labels"]):
            o = by_idx.get(i)
            if o is None:
                continue
            w_mm, d_mm, h_mm = o.width_m * 1000, o.depth_m * 1000, o.height_m * 1000
            # Always legacy clamp mode: n_corr/n_severe/prior_dev are defined
            # against the clamp reference regardless of the swept variant.
            sw, sd, sh, corrections = sanitize_dims(label, w_mm, d_mm, h_mm)
            bounded = label.lower().strip() in _BOUNDS
            dev = 0.0
            if bounded and min(w_mm, d_mm, h_mm) > 0:
                dev = float(
                    np.mean([
                        abs(math.log(max(a, 1e-6) / max(b, 1e-6)))
                        for a, b in ((w_mm, sw), (d_mm, sd), (h_mm, sh))
                    ])
                )
            # Predicted center depth (camera frame) vs observed bbox depth.
            center_cam = R.T @ np.asarray(o.center_world, dtype=np.float32)
            z_pred = float(center_cam[2])
            x1, y1, x2, y2 = [int(round(v)) for v in boxes[i]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(depth.shape[1], max(x2, x1 + 1)), min(depth.shape[0], max(y2, y1 + 1))
            patch = depth[y1:y2, x1:x2]
            patch = patch[patch > 1e-4]
            z_obs = float(np.median(patch)) if patch.size else float("nan")
            dlr = (
                math.log(z_pred / z_obs)
                if (z_pred > 1e-6 and z_obs and not math.isnan(z_obs) and z_obs > 1e-6)
                else float("nan")
            )
            records.append({
                "image": stem,
                "split": snap.get("split", "dev"),
                "det_index": i,
                "label": label,
                "score2d": float(snap["scores"][i]),
                "prob": float(o.confidence),
                "w_mm": round(w_mm, 1), "d_mm": round(d_mm, 1), "h_mm": round(h_mm, 1),
                "volume_m3": round(o.volume_m3, 6),
                "bounded": bounded,
                "n_corr": len(corrections),
                "n_severe": sum("severe" in c for c in corrections),
                "prior_dev": round(dev, 4),
                "z_pred": round(z_pred, 3),
                "z_obs": round(z_obs, 3) if not math.isnan(z_obs) else None,
                "depth_log_ratio": round(dlr, 4) if not math.isnan(dlr) else None,
            })
        all_records.extend(records)
        log.info("[run:%s] %s: %d objs (%.1fs)", args.name, stem, len(records), times[stem])

    ec.attach_manifest_gt(all_records, args.manifest)

    report = {
        "name": args.name,
        "lift_seconds": {k: round(v, 2) for k, v in times.items()},
        "records": all_records,
    }
    out = os.path.join(REPORT_DIR, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log.info("[run:%s] wrote %d records -> %s", args.name, len(all_records), out)
    return 0


# --------------------------------------------------------------------------- #
# Stage 3: compare variants (paired, vs first-listed baseline)
# --------------------------------------------------------------------------- #
def _agg(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    bounded = [r for r in records if r["bounded"]]
    dlrs = [abs(r["depth_log_ratio"]) for r in records if r["depth_log_ratio"] is not None]
    gt_errs = [r["gt_err"] for r in records if r.get("gt_err")]
    out = {
        "objects": len(records),
        "bounded": len(bounded),
        "axis_corr_rate": round(
            sum(r["n_corr"] for r in bounded) / max(1, 3 * len(bounded)), 4
        ),
        "obj_corr_rate": round(
            sum(1 for r in bounded if r["n_corr"]) / max(1, len(bounded)), 4
        ),
        "severe_rate": round(
            sum(r["n_severe"] for r in bounded) / max(1, 3 * len(bounded)), 4
        ),
        "mean_prior_dev": round(float(np.mean([r["prior_dev"] for r in bounded])), 4)
        if bounded else None,
        "mean_abs_depth_lr": round(float(np.mean(dlrs)), 4) if dlrs else None,
    }
    if gt_errs:
        # height may be absent (manifest height_mm: null) — average present axes.
        for axis in ("long", "short", "height"):
            vals = [e[axis] for e in gt_errs if axis in e]
            if vals:
                out[f"gt_mae_{axis}"] = round(float(np.mean(vals)), 4)
        out["gt_mae_all"] = round(
            float(np.mean([v for e in gt_errs for v in e.values()])), 4
        )
        out["gt_n"] = len(gt_errs)
    return out


def _paired(base: List[Dict[str, Any]], var: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    bmap = {(r["image"], r["det_index"]): r for r in base}
    deltas = []
    for r in var:
        b = bmap.get((r["image"], r["det_index"]))
        if b is None:
            continue
        bv, vv = b.get(key), r.get(key)
        if bv is None or vv is None:
            continue
        if key == "depth_log_ratio":
            bv, vv = abs(bv), abs(vv)
        deltas.append(vv - bv)
    if not deltas:
        return {"n": 0}
    pos = sum(1 for d in deltas if d > 1e-9)   # variant worse (metric is a badness)
    neg = sum(1 for d in deltas if d < -1e-9)  # variant better
    return {
        "n": len(deltas),
        "better": neg,
        "worse": pos,
        "ties": len(deltas) - pos - neg,
        "mean_delta": round(float(np.mean(deltas)), 4),
        "median_delta": round(float(np.median(deltas)), 4),
        "sign_p": round(ec.sign_test_p(pos, neg), 4),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    reports = {}
    for name in args.names:
        with open(os.path.join(REPORT_DIR, f"{name}.json"), "r", encoding="utf-8") as f:
            reports[name] = json.load(f)
    base_name = args.baseline or args.names[0]
    base = reports[base_name]["records"]

    print(f"\n=== aggregate (baseline: {base_name}) ===")
    rows = {}
    for name, rep in reports.items():
        rows[name] = {}
        for split in ("dev", "holdout", None):
            recs = rep["records"] if split is None else [
                r for r in rep["records"] if r["split"] == split
            ]
            if not recs:
                continue
            rows[name][split or "all"] = _agg(recs)
        total_t = sum(rep["lift_seconds"].values())
        rows[name]["lift_s_total"] = round(total_t, 1)
    print(json.dumps(rows, indent=1, ensure_ascii=False))

    print(f"\n=== paired deltas vs {base_name} (negative = variant better) ===")
    paired_out = {}
    for name, rep in reports.items():
        if name == base_name:
            continue
        paired_out[name] = {
            "prior_dev": _paired(base, rep["records"], "prior_dev"),
            "abs_depth_lr": _paired(base, rep["records"], "depth_log_ratio"),
        }
        if any(r.get("gt_err") for r in rep["records"]):
            bmap = {(r["image"], r["det_index"]): r for r in base}
            deltas = []
            for r in rep["records"]:
                b = bmap.get((r["image"], r["det_index"]))
                if not b or not r.get("gt_err") or not b.get("gt_err"):
                    continue
                ev = float(np.mean(list(r["gt_err"].values())))
                eb = float(np.mean(list(b["gt_err"].values())))
                deltas.append(ev - eb)
            if deltas:
                pos = sum(1 for d in deltas if d > 1e-9)
                neg = sum(1 for d in deltas if d < -1e-9)
                paired_out[name]["gt_err"] = {
                    "n": len(deltas), "better": neg, "worse": pos,
                    "mean_delta": round(float(np.mean(deltas)), 4),
                    "sign_p": round(ec.sign_test_p(pos, neg), 4),
                }
    print(json.dumps(paired_out, indent=1, ensure_ascii=False))

    out = os.path.join(REPORT_DIR, f"compare_{'_vs_'.join(args.names[:4])}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"aggregate": rows, "paired": paired_out}, f, ensure_ascii=False, indent=1)
    log.info("wrote %s", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="cache OWLv2 + Depth Pro outputs")
    sp.add_argument("--manifest", default=DEFAULT_MANIFEST)
    sp.add_argument("--only", default=None, help="comma-separated image stems")
    sp.set_defaults(fn=cmd_snapshot)

    rp = sub.add_parser("run", help="run a BoxerNet variant against the cache")
    rp.add_argument("--name", required=True)
    rp.add_argument("--manifest", default=DEFAULT_MANIFEST)
    rp.add_argument("--only", default=None)
    rp.set_defaults(fn=cmd_run)

    cp = sub.add_parser("compare", help="aggregate + paired comparison")
    cp.add_argument("names", nargs="+")
    cp.add_argument("--baseline", default=None)
    cp.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
