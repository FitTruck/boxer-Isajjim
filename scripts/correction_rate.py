"""Re-measure the sanitizer correction rate (opt.md A-4 follow-up).

Takes a fresh eval_ab report (real BoxerNet inference, raw dims) and computes,
per condition, how many objects/axes the sanitizer would correct:

  old-bounds + clamp   the original A-4 definition (89% bounded-object rate)
  new-bounds + clamp   after the washer/dryer/tv bounds corrections
  new-bounds + fused   same trigger condition; correction VALUES differ

The old bounds are reconstructed explicitly for the 5 entries edited on
2026-06-10 so the comparison is exact.

Usage:
  python scripts/correction_rate.py --report scripts/.eval_cache/reports/fresh_check.json
"""

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402  (repo bootstrap: sys.path, BOXER_* env, logging)

from ai.pipeline import dimension_bounds as db  # noqa: E402

# The exact pre-2026-06-10 values of the edited entries.
_OLD_ENTRIES = {
    "automatic washer": {"long": (550, 700), "short": (550, 700), "height": (800, 1000)},
    "washing machine": {"long": (550, 700), "short": (550, 700), "height": (800, 1000)},
    "washer dryer": {"long": (550, 720), "short": (550, 720), "height": (800, 1900)},
    "tv": {"long": (700, 1800), "short": (40, 150), "height": (400, 800)},
    "television set": {"long": (700, 1800), "short": (40, 150), "height": (400, 800)},
}


def _measure(records: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    bounded = corrected = severe_objs = axes_corr = 0
    by_class: Counter = Counter()
    for r in records:
        if r["label"].lower().strip() not in db._BOUNDS:
            continue
        bounded += 1
        _, _, _, corr = db.sanitize_dims(
            r["label"], r["w_mm"], r["d_mm"], r["h_mm"], prob=r["prob"], mode=mode
        )
        axis_corr = [c for c in corr if "reordered" not in c and "scale" not in c]
        if corr:
            corrected += 1
            by_class[r["label"]] += 1
        axes_corr += len(axis_corr)
        if any("severe" in c for c in corr):
            severe_objs += 1
    return {
        "bounded_objects": bounded,
        "obj_corr_rate": round(corrected / max(1, bounded), 3),
        "axis_corr_rate": round(axes_corr / max(1, 3 * bounded), 3),
        "severe_obj_rate": round(severe_objs / max(1, bounded), 3),
        "top_classes": by_class.most_common(6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--include-dup", action="store_true",
                    help="include 35.png (byte-identical dup of 31) like the A-4 measurement")
    args = ap.parse_args()

    report = ec.load_report(args.report)
    records = report["records"]
    if not args.include_dup:
        records = ec.drop_dup(records)

    print(f"report={report['name']}  objects={len(records)} "
          f"(dup35 {'included' if args.include_dup else 'excluded'})\n")

    # Condition 1: old bounds + clamp (A-4 definition)
    saved = {k: db._BOUNDS[k] for k in _OLD_ENTRIES}
    db._BOUNDS.update(_OLD_ENTRIES)
    try:
        old_clamp = _measure(records, "clamp")
    finally:
        db._BOUNDS.update(saved)

    new_clamp = _measure(records, "clamp")
    new_fused = _measure(records, "fused")

    for name, m in (("old-bounds+clamp (A-4 기준)", old_clamp),
                    ("new-bounds+clamp", new_clamp),
                    ("new-bounds+fused", new_fused)):
        print(f"[{name}]")
        print(f"  bounded objects      : {m['bounded_objects']}")
        print(f"  객체 보정률            : {m['obj_corr_rate']*100:.1f}%")
        print(f"  축 보정률             : {m['axis_corr_rate']*100:.1f}%")
        print(f"  severe 객체 비율       : {m['severe_obj_rate']*100:.1f}%")
        print(f"  최다 보정 클래스        : {m['top_classes']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
