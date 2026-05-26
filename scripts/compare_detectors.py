"""Head-to-head: OWLv2 vs SAM3 as the 2D front-end for BoxerNet.

For each image we run BOTH detectors, then lift each detector's boxes through
the SAME depth + BoxerNet so the only variable is the detector. Reports per
detector: latency, #detections, labels/confidence, and predicted WxDxH.

Run ON THE GPU SERVER (needs torch + OWLv2 + SAM3 + Depth Pro + BoxerNet):

    python scripts/compare_detectors.py path/to/image.jpg
    python scripts/compare_detectors.py --manifest scripts/accuracy_manifest.example.json

Note: SAM3 loads a second large model; expect higher VRAM. SAM3 latency scales
with the number of concepts (see ai/pipeline/sam3_detection.py).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
log = logging.getLogger("compare")


def _load_image(spec: Dict[str, Any], fetcher) -> Tuple[Optional[Any], str]:
    from PIL import Image

    if spec.get("path"):
        path = spec["path"]
        if not os.path.exists(path):
            return None, path
        return Image.open(path).convert("RGB"), path
    if spec.get("url"):
        url = spec["url"]
        return asyncio.run(fetcher.fetch_async(url)), url
    return None, "<no path/url>"


def _lift(boxer, image, det, depth_res) -> List[Dict[str, Any]]:
    """Lift one detector's boxes to 3D via BoxerNet; return per-object rows."""
    boxes = det["boxes"]
    labels = list(det["labels"])
    scores = list(det["scores"])
    if len(boxes) == 0:
        return []
    obbs = boxer.lift(
        image=image,
        bboxes_xyxy=boxes,
        labels=labels,
        depth=depth_res.depth,
        focal_length_px=depth_res.focal_length_px,
    )
    obb_by_idx = {obb.input_index: obb for obb in obbs}
    rows = []
    for i in range(len(boxes)):
        obb = obb_by_idx.get(i)
        rows.append(
            {
                "label": labels[i],
                "confidence": float(scores[i]),
                "wmm": round(obb.width_m * 1000.0, 0) if obb else 0.0,
                "dmm": round(obb.depth_m * 1000.0, 0) if obb else 0.0,
                "hmm": round(obb.height_m * 1000.0, 0) if obb else 0.0,
            }
        )
    return rows


def _print_side(name: str, latency_ms: float, rows: List[Dict[str, Any]]) -> None:
    print(f"\n  --- {name}: {len(rows)} detections, detect={latency_ms:.0f}ms ---")
    for r in sorted(rows, key=lambda x: -x["confidence"]):
        print(
            f"    {r['label']:<22} conf={r['confidence']:.2f} "
            f"WxDxH={r['wmm']:.0f}x{r['dmm']:.0f}x{r['hmm']:.0f}mm"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="OWLv2 vs SAM3 detector comparison for BoxerNet")
    ap.add_argument("image", nargs="?", help="single image path")
    ap.add_argument("--manifest", help="JSON manifest with images (path/url)")
    args = ap.parse_args()
    if not args.image and not args.manifest:
        ap.error("provide an image path or --manifest")

    from ai.config import Config
    from ai.pipeline import BoxerLifter, DepthEstimator, ImageFetcher, Owlv2Detector, Sam3Detector

    log.info("device=%s depth=%s", Config.get_default_device(), Config.DEPTH_BACKEND)
    fetcher = ImageFetcher()
    depth = DepthEstimator()
    boxer = BoxerLifter()
    log.info("Loading OWLv2 ...")
    owlv2 = Owlv2Detector()
    log.info("Loading SAM3 ...")
    sam3 = Sam3Detector()

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            specs = json.load(f).get("images", [])
    else:
        specs = [{"path": args.image}]

    totals = {"OWLv2": {"n": 0, "ms": 0.0}, "SAM3": {"n": 0, "ms": 0.0}}
    for spec in specs:
        image, source = _load_image(spec, fetcher)
        if image is None:
            log.error("could not load: %s", source)
            continue
        depth_res = depth.estimate(image)  # shared — detector is the only variable
        print(f"\n===== {source}  ({image.size[0]}x{image.size[1]})  "
              f"focal_px={depth_res.focal_length_px} =====")

        for name, det_obj in (("OWLv2", owlv2), ("SAM3", sam3)):
            t = time.perf_counter()
            det = det_obj.detect(image)
            ms = (time.perf_counter() - t) * 1000.0
            rows = _lift(boxer, image, det, depth_res)
            _print_side(name, ms, rows)
            totals[name]["n"] += len(rows)
            totals[name]["ms"] += ms

    print("\n========== SUMMARY ==========")
    for name in ("OWLv2", "SAM3"):
        t = totals[name]
        print(f"  {name:<6}: {t['n']} total detections, {t['ms']:.0f}ms total detect time")


if __name__ == "__main__":
    main()
