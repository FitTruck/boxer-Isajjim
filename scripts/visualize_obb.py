"""Run the full Boxer pipeline on one image and render absolute-dimension 3D boxes.

Loads OWLv2 → Depth Pro → BoxerNet (the real production pipeline), then projects
each detection's 3D oriented bounding box (with its absolute metric W×D×H) back
onto the image as a wireframe cuboid via Boxer's own `draw_bb3s` renderer.

Usage:
    python scripts/visualize_obb.py [image_path] [-o out.png] [--conf 0.2]

Outputs:
    - an annotated PNG (default: <image_stem>_boxes.png next to the input)
    - a per-object table of absolute dimensions (mm), volume (m³), confidence.
"""

import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image

# Make repo root importable + supply Boxer checkpoint defaults (mirrors e2e_pipeline).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402  (repo bootstrap: sys.path, BOXER_* env, logging)

_ROOT = ec.ROOT

log = logging.getLogger("viz_obb")

# Distinct BGR colors cycled per object (cv2 draws in BGR).
_PALETTE = [
    (66, 135, 245), (60, 200, 80), (40, 40, 230), (0, 200, 255),
    (230, 130, 40), (200, 60, 200), (180, 200, 50), (50, 160, 240),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", default=os.path.join(_ROOT, "ai", "imgs", "35.png"))
    ap.add_argument("-o", "--out", default=None, help="output PNG path")
    ap.add_argument("--conf", type=float, default=0.2, help="OBB confidence threshold")
    ap.add_argument("--top", type=int, default=0,
                    help="draw only the N largest boxes by volume (0 = all)")
    ap.add_argument("--height", type=int, default=900,
                    help="output image height in px (keeps source aspect ratio)")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        log.error("Image not found: %s", args.image)
        return 2
    out_path = args.out or os.path.splitext(args.image)[0] + "_boxes.png"

    # Imports deferred so env defaults above are in effect first.
    import cv2

    from ai.config import Config
    from ai.pipeline import BoxerLifter, DepthEstimator, Owlv2Detector
    from ai.pipeline.furniture_pipeline import _dims_mm

    log.info("Device=%s  depth_backend=%s", Config.get_default_device(), Config.DEPTH_BACKEND)
    log.info("Boxer ckpt=%s (exists=%s)", Config.BOXER_CHECKPOINT,
             os.path.exists(Config.BOXER_CHECKPOINT or ""))

    img = Image.open(args.image).convert("RGB")
    W0, H0 = img.size
    log.info("Loaded %s  size=%dx%d", args.image, W0, H0)

    log.info("Loading models (OWLv2 + Depth + BoxerNet)...")
    detector = Owlv2Detector()
    depth_model = DepthEstimator()
    boxer = BoxerLifter(conf_threshold=args.conf)
    if boxer.net is None:
        log.error("BoxerNet did not load — check BOXER_CHECKPOINT / BOXER_REPO_PATH.")
        return 3

    # Stage 1: 2D detection.
    det = detector.detect(img)
    boxes, labels = det["boxes"], list(det["labels"])
    log.info("OWLv2: %d detections", len(boxes))
    if len(boxes) == 0:
        log.warning("No detections — nothing to lift.")
        return 0

    # Stage 2: metric depth (+ focal).
    depth_result = depth_model.estimate(img)
    log.info("Depth: focal_px=%s", depth_result.focal_length_px)

    # Stage 3: 3D OBB lift — raw ObbTW + the exact camera/pose used for inference.
    lifted = boxer.lift_obbs(
        image=img,
        bboxes_xyxy=boxes,
        depth=depth_result.depth,
        focal_length_px=depth_result.focal_length_px,
    )
    if lifted is None:
        log.error("lift_obbs returned None.")
        return 3
    obb_w, cam, pose, img_t = lifted

    # Confident subset (same filter the production pipeline applies), aligned to
    # the original detections via BoxerObb.input_index.
    results = boxer._to_results(obb_w, labels)
    if not results:
        log.warning("No OBB above conf=%.2f.", args.conf)
    keep_idx = [o.input_index for o in results]

    # Absolute metric dimensions (mm, with the pipeline's per-class sanitizer).
    # `rows[k]` aligns with `results[k]` and `keep_idx[k]`.
    rows = []
    for o in results:
        w_mm, d_mm, h_mm, vol_m3, corr = _dims_mm(o.label, o)
        rows.append((o.label, w_mm, d_mm, h_mm, vol_m3, o.confidence, o.center_world, corr))

    # Draw the largest boxes first (front-most labels win); optionally keep only
    # the top-N by volume to declutter. `order` indexes into results/rows.
    order = sorted(range(len(rows)), key=lambda k: rows[k][4], reverse=True)
    if args.top > 0:
        order = order[: args.top]
    texts = [f"{k}:{rows[k][0]} {rows[k][1]:.0f}x{rows[k][2]:.0f}x{rows[k][3]:.0f}mm"
             for k in order]
    colors = [_PALETTE[k % len(_PALETTE)] for k in order]

    # Render: project the 3D wireframes onto the 960x960 inference image, then
    # un-squish to the source aspect ratio at the requested height (legible text).
    from utils.image import draw_bb3s, torch2cv2  # boxer repo (on sys.path after load)

    viz = np.ascontiguousarray(torch2cv2(img_t))  # HxWxC BGR uint8 (960x960)
    # draw_bb3s expects an UNBATCHED camera + pose (CameraTW.from_surreal already
    # squeezes the camera; the pose is built as (1,12), so drop its batch dim).
    pose_draw = pose[0] if pose.ndim == 2 else pose
    if order:
        obb_draw = obb_w[[keep_idx[k] for k in order]]
        viz = draw_bb3s(
            viz, pose_draw, cam, obb_draw,
            draw_bb3_center=True, draw_label=True, draw_score=False,
            texts=texts, colors=colors, rotate_label=False,
            thickness=2, text_sz=0.5,
        )
    out_h = max(1, args.height)
    out_w = max(1, round(W0 * out_h / H0))
    viz = cv2.resize(viz, (out_w, out_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, viz)

    # Report.
    print("\n=========== ABSOLUTE-DIMENSION BOXES ===========")
    print(f"image : {args.image}  ({W0}x{H0})")
    print(f"output: {out_path}")
    drawn = set(order)
    note = f"  (drawing top {args.top} by volume)" if args.top > 0 else ""
    print(f"objects above conf={args.conf}: {len(rows)}{note}\n")
    print(f"{'#':>2} {'D':>1} {'label':<20} {'W×D×H (mm)':>22}  {'vol(m³)':>8}  {'conf':>5}  center_world(x,y,z m)")
    print("-" * 94)
    for i, (label, w, d, h, vol, conf, ctr, corr) in enumerate(rows):
        cx, cy, cz = ctr
        flag = "  *sanitized" if corr else ""
        mark = "•" if i in drawn else " "
        print(f"{i:>2} {mark:>1} {label:<20} {w:>6.0f}×{d:>6.0f}×{h:>6.0f}  {vol:>8.3f}  "
              f"{conf:>5.2f}  ({cx:+.2f},{cy:+.2f},{cz:+.2f}){flag}")
    print("=" * 94)
    print("D = drawn in image (label prefix '#:').\n")
    log.info("Saved annotated image -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
