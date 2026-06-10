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
  python scripts/eval_ab.py run --name native50k --sdp-source native --sdp-points 57600
  python scripts/eval_ab.py compare baseline nearest native14k native50k resized50k
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault(
    "BOXER_CHECKPOINT",
    os.path.join(_ROOT, "boxer", "ckpts", "boxernet_hw960in4x6d768-3e37cfc4.ckpt"),
)
os.environ.setdefault("BOXER_REPO_PATH", os.path.join(_ROOT, "boxer"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("eval_ab")

CACHE_DIR = os.path.join(_ROOT, "scripts", ".eval_cache")
SNAP_DIR = os.path.join(CACHE_DIR, "snapshot")
REPORT_DIR = os.path.join(CACHE_DIR, "reports")
DEFAULT_MANIFEST = os.path.join(_ROOT, "scripts", "eval_manifest.json")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data["images"]
    for item in images:
        if not os.path.isabs(item["path"]):
            item["path"] = os.path.join(_ROOT, item["path"])
    return images


def _filter_only(images: List[Dict[str, Any]], only: Optional[str]) -> List[Dict[str, Any]]:
    if not only:
        return images
    keep = {s.strip() for s in only.split(",") if s.strip()}
    return [im for im in images if _stem(im["path"]) in keep]


def sign_test_p(pos: int, neg: int) -> float:
    """Exact two-sided binomial sign test (p=0.5), zero deltas excluded upstream."""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------- #
# Stage 1: snapshot (OWLv2 + Depth Pro, cached once)
# --------------------------------------------------------------------------- #
def cmd_snapshot(args: argparse.Namespace) -> int:
    from PIL import Image

    from ai.pipeline import DepthEstimator, Owlv2Detector

    images = _filter_only(_load_manifest(args.manifest), args.only)
    os.makedirs(SNAP_DIR, exist_ok=True)

    detector = Owlv2Detector()
    depth_model = DepthEstimator()

    for item in images:
        path, stem = item["path"], _stem(item["path"])
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
def _match_gt(
    records: List[Dict[str, Any]], gt_objects: List[Dict[str, Any]]
) -> None:
    """Greedy label match: highest-2D-score detection takes the first GT row."""
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for gt in gt_objects:
        by_label.setdefault(gt["label"].lower().strip(), []).append(gt)
    for rec in sorted(records, key=lambda r: -r["score2d"]):
        pool = by_label.get(rec["label"].lower().strip())
        if not pool:
            continue
        gt = pool.pop(0)
        # height_mm may be null (axis not pinned by the standard) -> skip axis.
        rec["gt"] = {
            k: (float(gt[k]) if gt.get(k) is not None else None)
            for k in ("width_mm", "depth_mm", "height_mm")
        }
        if min(rec["w_mm"], rec["d_mm"], rec["h_mm"]) <= 0:
            continue
        p_long, p_short = max(rec["w_mm"], rec["d_mm"]), min(rec["w_mm"], rec["d_mm"])
        g_long = max(gt["width_mm"], gt["depth_mm"])
        g_short = min(gt["width_mm"], gt["depth_mm"])
        if g_long <= 0 or g_short <= 0:
            continue
        errs = {
            "long": abs(math.log(p_long / g_long)),
            "short": abs(math.log(p_short / g_short)),
        }
        g_h = gt.get("height_mm")
        if g_h:
            errs["height"] = abs(math.log(rec["h_mm"] / g_h))
        rec["gt_err"] = errs


def cmd_run(args: argparse.Namespace) -> int:
    from PIL import Image

    from ai.pipeline import BoxerLifter
    from ai.pipeline.dimension_bounds import _BOUNDS, sanitize_dims

    images = _filter_only(_load_manifest(args.manifest), args.only)
    os.makedirs(REPORT_DIR, exist_ok=True)

    lifter = BoxerLifter(
        conf_threshold=0.0,
        sdp_source=args.sdp_source,
        sdp_interp=args.sdp_interp,
        sdp_target_points=args.sdp_points,
    )
    if lifter.net is None:
        log.error("BoxerNet unavailable (checkpoint/repo missing) — abort")
        return 2
    # Camera->world default rotation used to read back the predicted center
    # depth in the camera frame (harness runs the level-camera default path).
    R = BoxerLifter._R_WORLD_FROM_CAM

    all_records: List[Dict[str, Any]] = []
    times: Dict[str, float] = {}
    for item in images:
        stem = _stem(item["path"])
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
        if item.get("objects"):
            _match_gt(records, item["objects"])
        all_records.extend(records)
        log.info("[run:%s] %s: %d objs (%.1fs)", args.name, stem, len(records), times[stem])

    report = {
        "name": args.name,
        "params": {
            "sdp_source": args.sdp_source,
            "sdp_interp": args.sdp_interp,
            "sdp_points": args.sdp_points,
        },
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
        for axis in ("long", "short", "height"):
            out[f"gt_mae_{axis}"] = round(float(np.mean([e[axis] for e in gt_errs])), 4)
        out["gt_mae_all"] = round(
            float(np.mean([e[a] for e in gt_errs for a in ("long", "short", "height")])), 4
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
        "sign_p": round(sign_test_p(pos, neg), 4),
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
                    "sign_p": round(sign_test_p(pos, neg), 4),
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
    rp.add_argument("--sdp-source", default="resized", choices=["resized", "native"])
    rp.add_argument("--sdp-interp", default="bilinear", choices=["bilinear", "nearest"])
    rp.add_argument("--sdp-points", type=int, default=14400)
    rp.set_defaults(fn=cmd_run)

    cp = sub.add_parser("compare", help="aggregate + paired comparison")
    cp.add_argument("names", nargs="+")
    cp.add_argument("--baseline", default=None)
    cp.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
