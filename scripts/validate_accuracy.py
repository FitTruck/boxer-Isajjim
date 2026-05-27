"""Accuracy validation harness — isolate the dimension-error source.

The production callback only returns final WxDxH, which hides *where* an error
comes from. This tool runs the full pipeline per image while exposing the
intermediate signals:

  - Depth Pro `focal_length_px` and a plausibility ratio `focal / img_width`
    (a normal phone photo lands around 1.0-1.5; wildly off => focal is suspect).
  - Median scene depth (meters) inside each detected bbox — sanity-check the
    metric scale (e.g. a sofa photographed across a room should be ~2-4 m).
  - Predicted WxDxH per object.
  - If ground-truth dims are supplied, per-dimension error % AND a global
    multiplicative scale bias = geomean(pred / gt) over all matched dims.

Reading the scale bias:
  - bias ~= 1.0            -> metric scale is fine; residual error is per-object
                             geometry (BoxerNet / bbox quality).
  - bias consistently != 1 -> every dimension is off by the same factor, which
                             is the fingerprint of a depth/focal SCALE error,
                             not a BoxerNet geometry error. Swapping the depth
                             model (e.g. UniDepthV2) is then justified.

Run ON THE GPU SERVER (needs torch + the 3 models + BOXER_CHECKPOINT):

    # quick single-image diagnostic (no ground truth)
    python scripts/validate_accuracy.py path/to/image.jpg

    # batch + ground-truth comparison
    python scripts/validate_accuracy.py --manifest manifest.json [--report out.json]

Manifest JSON:
    {
      "images": [
        {"path": "/abs/img1.jpg",
         "objects": [
           {"label": "sofa", "width_mm": 1850, "depth_mm": 900, "height_mm": 800}
         ]},
        {"url": "https://.../img2.jpg"}      # GT optional; omit "objects" for diagnostic only
      ]
    }
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- Make repo root importable + default Boxer paths (same as tests/e2e_pipeline.py) ---
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
log = logging.getLogger("validate")

_DIMS = ("width_mm", "depth_mm", "height_mm")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundTruth:
    label: str
    width_mm: float
    depth_mm: float
    height_mm: float


@dataclass
class ObjectReport:
    label: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_depth_median_m: Optional[float]
    pred: Dict[str, float]                       # width_mm/depth_mm/height_mm
    gt: Optional[Dict[str, float]] = None
    err_pct: Dict[str, float] = field(default_factory=dict)


@dataclass
class ImageReport:
    source: str
    size: Tuple[int, int]
    focal_length_px: Optional[float]
    focal_over_width: Optional[float]
    objects: List[ObjectReport] = field(default_factory=list)
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Image loading (local path or URL via the pipeline's fetcher)
# --------------------------------------------------------------------------- #
def _load_image(spec: Dict[str, Any], fetcher) -> Tuple[Optional[Any], str]:
    from PIL import Image

    if spec.get("path"):
        path = spec["path"]
        if not os.path.exists(path):
            return None, path
        return Image.open(path).convert("RGB"), path
    if spec.get("url"):
        url = spec["url"]
        img = asyncio.run(fetcher.fetch_async(url))
        return img, url
    return None, "<no path/url>"


# --------------------------------------------------------------------------- #
# Ground-truth matching: greedily pair each GT to the best unused prediction of
# the same label (case-insensitive); fall back to confidence order.
# --------------------------------------------------------------------------- #
def _match(objs: List[ObjectReport], gts: List[GroundTruth]) -> None:
    used: set[int] = set()
    for gt in gts:
        best_i, best_conf = None, -1.0
        for i, o in enumerate(objs):
            if i in used:
                continue
            same = gt.label.lower() in o.label.lower() or o.label.lower() in gt.label.lower()
            if same and o.confidence > best_conf:
                best_i, best_conf = i, o.confidence
        if best_i is None:  # no label match -> take highest-confidence unused
            for i, o in enumerate(objs):
                if i not in used and o.confidence > best_conf:
                    best_i, best_conf = i, o.confidence
        if best_i is None:
            log.warning("  GT %s: no prediction to match", gt.label)
            continue
        used.add(best_i)
        o = objs[best_i]
        o.gt = {"width_mm": gt.width_mm, "depth_mm": gt.depth_mm, "height_mm": gt.height_mm}
        for d in _DIMS:
            if o.gt[d]:
                o.err_pct[d] = (o.pred[d] - o.gt[d]) / o.gt[d] * 100.0


# --------------------------------------------------------------------------- #
# Per-image run: drive stages manually so we can capture focal + depth.
# --------------------------------------------------------------------------- #
def _run_image(pipe, image, source: str, grav_est=None) -> ImageReport:
    import numpy as np

    w, h = image.size
    det = pipe.detector.detect(image)
    boxes = det["boxes"]
    labels = list(det["labels"])
    scores = list(det["scores"])

    depth_res = pipe.depth_model.estimate(image)
    fl = depth_res.focal_length_px
    gravity = None
    if grav_est is not None:
        cg = grav_est.estimate(image)
        # GeoCalib focal is purpose-built and more reliable; use it + its gravity.
        fl = cg.focal_px
        gravity = cg.gravity_down_cam
        log.info(
            "  GeoCalib: focal=%.0f roll=%.1f pitch=%.1f deg gravity_down=%s",
            cg.focal_px, cg.roll_deg, cg.pitch_deg,
            np.round(cg.gravity_down_cam, 3).tolist(),
        )

    rep = ImageReport(
        source=source,
        size=(w, h),
        focal_length_px=float(fl) if fl is not None else None,
        focal_over_width=(float(fl) / w) if fl else None,
    )

    obbs = pipe.boxer.lift(
        image=image,
        bboxes_xyxy=boxes,
        labels=labels,
        depth=depth_res.depth,
        focal_length_px=fl,
        gravity_down_cam=gravity,
    )
    obb_by_idx = {obb.input_index: obb for obb in obbs}

    depth = depth_res.depth
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        region = depth[y1:y2, x1:x2]
        med = float(np.median(region)) if region.size else None

        obb = obb_by_idx.get(i)
        if obb:
            w_mm, d_mm, h_mm = obb.width_m * 1000.0, obb.depth_m * 1000.0, obb.height_m * 1000.0
            from ai.config import Config

            if Config.SANITIZE_DIMENSIONS:
                from ai.pipeline.dimension_bounds import sanitize_dims

                w_mm, d_mm, h_mm, corr = sanitize_dims(labels[i], w_mm, d_mm, h_mm)
                for c in corr:
                    log.info("  [sanitize] %s", c)
        else:
            w_mm = d_mm = h_mm = 0.0
        pred = {
            "width_mm": round(w_mm, 1),
            "depth_mm": round(d_mm, 1),
            "height_mm": round(h_mm, 1),
        }
        rep.objects.append(
            ObjectReport(
                label=labels[i],
                confidence=float(scores[i]),
                bbox_xyxy=[float(v) for v in box],
                bbox_depth_median_m=round(med, 3) if med is not None else None,
                pred=pred,
            )
        )
    return rep


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_image(rep: ImageReport) -> None:
    print(f"\n===== {rep.source}  ({rep.size[0]}x{rep.size[1]}) =====")
    if rep.error:
        print(f"  ERROR: {rep.error}")
        return
    fr = f"{rep.focal_over_width:.2f}" if rep.focal_over_width else "n/a"
    flag = ""
    if rep.focal_over_width and not (0.6 <= rep.focal_over_width <= 2.0):
        flag = "  <-- focal/width out of plausible 0.6-2.0 range!"
    print(f"  focal_px={rep.focal_length_px}  focal/width={fr}{flag}")
    for o in rep.objects:
        line = (
            f"  {o.label:<20} conf={o.confidence:.2f} "
            f"depth_med={o.bbox_depth_median_m}m  "
            f"WxDxH={o.pred['width_mm']:.0f}x{o.pred['depth_mm']:.0f}x{o.pred['height_mm']:.0f}mm"
        )
        if o.gt:
            errs = " ".join(f"{d.split('_')[0]}={o.err_pct.get(d, 0):+.0f}%" for d in _DIMS)
            line += f"  | GT {o.gt['width_mm']:.0f}x{o.gt['depth_mm']:.0f}x{o.gt['height_mm']:.0f}  err[{errs}]"
        print(line)


def _aggregate(reports: List[ImageReport]) -> None:
    ratios: List[float] = []
    abs_pct: Dict[str, List[float]] = {d: [] for d in _DIMS}
    matched = 0
    for rep in reports:
        for o in rep.objects:
            if not o.gt:
                continue
            matched += 1
            for d in _DIMS:
                if o.gt[d] and o.pred[d] > 0:
                    ratios.append(o.pred[d] / o.gt[d])
                    abs_pct[d].append(abs(o.err_pct.get(d, 0.0)))

    print("\n========== AGGREGATE ==========")
    if matched == 0:
        print("  No ground-truth matches — diagnostic mode only.")
        print("  Sanity-check focal/width (~1.0-1.5) and bbox depth_med (plausible meters) above.")
        return

    print(f"  matched objects: {matched}")
    for d in _DIMS:
        vals = abs_pct[d]
        if vals:
            print(f"  MAPE {d:<9}: {sum(vals) / len(vals):.1f}%")
    if ratios:
        bias = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"  GLOBAL SCALE BIAS (geomean pred/gt): {bias:.3f}")
        if 0.9 <= bias <= 1.1:
            print("  => scale is good; residual error is per-object geometry (bbox/BoxerNet).")
        else:
            pct = (bias - 1.0) * 100.0
            print(f"  => every dim is ~{pct:+.0f}% off by a CONSISTENT factor.")
            print("     This is a depth/focal SCALE fingerprint, not BoxerNet geometry.")
            print("     Check focal_px plausibility & bbox depth_med; a depth-model swap")
            print("     (e.g. UniDepthV2) is justified.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_manifest(path: str) -> List[Tuple[Dict[str, Any], List[GroundTruth]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for spec in data.get("images", []):
        gts = [GroundTruth(**g) for g in spec.get("objects", [])]
        out.append((spec, gts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline accuracy / error-source validation")
    ap.add_argument("image", nargs="?", help="single image path for quick diagnostic")
    ap.add_argument("--manifest", help="JSON manifest with images + optional ground-truth dims")
    ap.add_argument("--report", help="write full JSON report to this path")
    ap.add_argument(
        "--geocalib",
        action="store_true",
        help="estimate per-image gravity + focal with GeoCalib and feed them to BoxerNet",
    )
    args = ap.parse_args()

    if not args.image and not args.manifest:
        ap.error("provide an image path or --manifest")

    from ai.config import Config
    from ai.pipeline import FurniturePipeline

    log.info("Device: %s | Depth backend: %s", Config.get_default_device(), Config.DEPTH_BACKEND)
    log.info("Boxer ckpt: %s (exists=%s)", Config.BOXER_CHECKPOINT,
             os.path.exists(Config.BOXER_CHECKPOINT or ""))
    t0 = time.time()
    pipe = FurniturePipeline(enable_3d=True)
    log.info("Pipeline ready in %.1fs", time.time() - t0)

    grav_est = None
    if args.geocalib:
        from ai.pipeline.gravity_estimation import CameraGravityEstimator

        grav_est = CameraGravityEstimator()

    if args.manifest:
        jobs = _parse_manifest(args.manifest)
    else:
        jobs = [({"path": args.image}, [])]

    reports: List[ImageReport] = []
    for spec, gts in jobs:
        image, source = _load_image(spec, pipe.fetcher)
        if image is None:
            log.error("could not load image: %s", source)
            reports.append(ImageReport(source=source, size=(0, 0),
                                       focal_length_px=None, focal_over_width=None,
                                       error="load failed"))
            continue
        rep = _run_image(pipe, image, source, grav_est=grav_est)
        if gts:
            _match(rep.objects, gts)
        _print_image(rep)
        reports.append(rep)

    _aggregate(reports)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "source": r.source, "size": r.size,
                        "focal_length_px": r.focal_length_px,
                        "focal_over_width": r.focal_over_width,
                        "error": r.error,
                        "objects": [vars(o) for o in r.objects],
                    }
                    for r in reports
                ],
                f, indent=2, ensure_ascii=False,
            )
        log.info("report written: %s", args.report)


if __name__ == "__main__":
    main()
