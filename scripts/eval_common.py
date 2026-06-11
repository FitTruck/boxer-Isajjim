"""Shared bootstrap + helpers for the eval/diagnostic scripts in this folder.

Import this FIRST (before any ``ai.*`` import): it makes the repo root
importable, points BOXER_CHECKPOINT / BOXER_REPO_PATH at the default local
clone, and configures logging — previously copy-pasted into every script.

Scripts are run as ``python scripts/<x>.py`` — scripts/ is then sys.path[0]
so ``import eval_common`` resolves. ``python -m scripts.x`` is not supported.
"""

import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault(
    "BOXER_CHECKPOINT",
    os.path.join(ROOT, "boxer", "ckpts", "boxernet_hw960in4x6d768-3e37cfc4.ckpt"),
)
os.environ.setdefault("BOXER_REPO_PATH", os.path.join(ROOT, "boxer"))
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

CACHE_DIR = os.path.join(ROOT, "scripts", ".eval_cache")
SNAP_DIR = os.path.join(CACHE_DIR, "snapshot")
REPORT_DIR = os.path.join(CACHE_DIR, "reports")
DEFAULT_MANIFEST = os.path.join(ROOT, "scripts", "eval_manifest.json")

#: ai/imgs/35.png is byte-identical to 31.png (md5 43e855be) — statistics must
#: not double-count it.
EXCLUDE_IMAGES = {"35"}


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_report(name_or_path: str) -> Dict[str, Any]:
    """Load a variant report by bare name (resolved in REPORT_DIR) or by path."""
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(REPORT_DIR, f"{name_or_path}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: str = DEFAULT_MANIFEST) -> List[Dict[str, Any]]:
    """Manifest images with paths resolved absolute against the repo root."""
    with open(path, encoding="utf-8") as f:
        images = json.load(f)["images"]
    for item in images:
        if not os.path.isabs(item["path"]):
            item["path"] = os.path.join(ROOT, item["path"])
    return images


def drop_dup(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r["image"] not in EXCLUDE_IMAGES]


def axis_log_errors(
    w_mm: float, d_mm: float, h_mm: float, gt: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """Per-axis |log(pred/gt)| with the footprint long/short aligned.

    height is scored only when the GT height is truthy (the manifest uses
    ``height_mm: null`` for axes not pinned by the standard). Returns None
    when the prediction or GT footprint is degenerate (<= 0).
    """
    g_long = max(gt["width_mm"], gt["depth_mm"])
    g_short = min(gt["width_mm"], gt["depth_mm"])
    p_long, p_short = max(w_mm, d_mm), min(w_mm, d_mm)
    if min(p_long, p_short) <= 0 or min(g_long, g_short) <= 0:
        return None
    errs = {
        "long": abs(math.log(p_long / g_long)),
        "short": abs(math.log(p_short / g_short)),
    }
    g_h = gt.get("height_mm")
    if g_h and h_mm > 0:
        errs["height"] = abs(math.log(h_mm / g_h))
    return errs


def attach_manifest_gt(
    records: List[Dict[str, Any]], manifest_path: str = DEFAULT_MANIFEST
) -> None:
    """Match manifest GT anchors onto records (greedy 1:1 by 2D score).

    Each GT row is consumed by the highest-scoring unmatched detection of the
    same label in the same image. Sets ``rec["gt"]`` (height may be None) and
    ``rec["gt_err"]`` (per-axis |log| errors) when computable.
    """
    for im in load_manifest(manifest_path):
        if not im.get("objects"):
            continue
        st = stem(im["path"])
        img_recs = sorted(
            (r for r in records if r["image"] == st),
            key=lambda r: -r["score2d"],
        )
        pool: Dict[str, List[Dict[str, Any]]] = {}
        for gt in im["objects"]:
            pool.setdefault(gt["label"].lower().strip(), []).append(gt)
        for rec in img_recs:
            rows = pool.get(rec["label"].lower().strip())
            if not rows:
                continue
            gt = rows.pop(0)
            rec["gt"] = {
                k: (float(gt[k]) if gt.get(k) is not None else None)
                for k in ("width_mm", "depth_mm", "height_mm")
            }
            errs = axis_log_errors(rec["w_mm"], rec["d_mm"], rec["h_mm"], rec["gt"])
            if errs:
                rec["gt_err"] = errs


def sign_test_p(pos: int, neg: int) -> float:
    """Exact two-sided binomial sign test (p=0.5); zero deltas excluded upstream."""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)
