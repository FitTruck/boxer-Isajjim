"""End-to-end smoke driver: OWLv2 → Depth → Boxer on a local image.

Usage:
    python tests/e2e_pipeline.py [image_path]
"""

import json
import logging
import os
import sys
import time

# Make repo root importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Default Boxer checkpoint if user didn't export
os.environ.setdefault(
    "BOXER_CHECKPOINT",
    os.path.join(_ROOT, "boxer", "ckpts", "boxernet_hw960in4x6d768-3e37cfc4.ckpt"),
)
os.environ.setdefault("BOXER_REPO_PATH", os.path.join(_ROOT, "boxer"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("e2e")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _ROOT, "boxer", "docs", "images", "sunrgbd_screenshot.jpg"
    )
    if not os.path.exists(image_path):
        log.error("Image not found: %s", image_path)
        sys.exit(2)

    from PIL import Image

    from ai.config import Config
    from ai.pipeline import FurniturePipeline

    log.info("Device: %s", Config.get_default_device())
    log.info("Devices pool: %s", Config.get_available_devices())
    log.info("Depth backend: %s", Config.DEPTH_BACKEND)
    log.info("Boxer ckpt: %s (exists=%s)", Config.BOXER_CHECKPOINT,
             os.path.exists(Config.BOXER_CHECKPOINT or ""))

    t0 = time.time()
    log.info("Initializing FurniturePipeline (loading all 3 models)...")
    pipe = FurniturePipeline(enable_3d=True)
    log.info("Pipeline ready in %.1fs", time.time() - t0)

    log.info("Loading test image: %s", image_path)
    img = Image.open(image_path).convert("RGB")
    log.info("Image size: %s", img.size)

    t1 = time.time()
    result = pipe.process_pil(img, image_id=1, image_url=image_path)
    log.info("Inference done in %.2fs (objects=%d)", time.time() - t1,
             len(result.objects))

    payload = FurniturePipeline.to_json_response([result])
    print("\n========== E2E RESULT (JSON) ==========")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=======================================\n")

    print("---- Per-object detail ----")
    for o in result.objects:
        print(
            f"  label={o.label:<22} "
            f"conf={o.confidence:.3f} "
            f"WxDxH={o.width_mm:.0f}x{o.depth_mm:.0f}x{o.height_mm:.0f} mm "
            f"vol={o.volume_m3:.3f} m³ "
            f"center=({o.center_xy[0]:.0f},{o.center_xy[1]:.0f})"
        )

    if not result.objects:
        print("(no objects detected — try a clearer interior photo)")
    print()
    print("E2E completed successfully.")


if __name__ == "__main__":
    main()
