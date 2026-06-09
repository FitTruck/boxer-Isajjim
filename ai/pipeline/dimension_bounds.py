"""Per-class physical sanity bounds for furniture dimensions (mm).

BoxerNet occasionally produces physically impossible dimensions (e.g. a bed
lifted to 2.4 m tall) on hard monocular scenes. Since the detector taxonomy is
fixed, we clamp each output dimension to a plausible per-class range:

  - in range            -> keep the model value (it may be correct)
  - mildly out of range -> clamp to the nearest bound (minimal correction)
  - severely out (< 0.5*min or > 2*max) -> replace with the class-typical value
    (midpoint); a value this far off means the box is mis-oriented, so the prior
    beats a clamped boundary.

Horizontal axes (width/depth) can swap, so they are bounded as a *footprint*:
the larger horizontal dim is checked against `long`, the smaller against
`short`, then mapped back. Height (vertical/gravity) is unambiguous.

For a moving-volume use case, class dimension priors are a legitimate signal,
not just a band-aid; corrections are logged so the rate can be monitored.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Range = Tuple[float, float]

# (long horizontal, short horizontal, height) ranges in mm. Rough, tunable seeds.
# Covers the full OWLv2 LVIS+ vocabulary (lvisplus_classes.csv) plus SAM3 concept
# aliases. Genuinely non-cuboidal classes are intentionally omitted (see _SKIP).
_BOUNDS: Dict[str, Dict[str, Range]] = {
    # ---- Beds / sleeping ----
    # KR mattress standard: length ~2000mm; width S 1000 / SS 1100 / Q 1500 /
    # K 1600-1650 / LK 1700 (+frame). "bed" box is mattress/frame (low); a tall
    # headboard is its own class.
    "bed": {"long": (1950, 2150), "short": (1000, 1900), "height": (200, 700)},
    "bunk bed": {"long": (1950, 2150), "short": (1000, 1300), "height": (1400, 2000)},
    "crib": {"long": (1100, 1400), "short": (600, 800), "height": (800, 1100)},
    "mattress": {"long": (1950, 2100), "short": (1000, 1800), "height": (150, 400)},
    "headboard": {"long": (1200, 2200), "short": (40, 200), "height": (400, 1300)},
    "pillow": {"long": (300, 800), "short": (300, 700), "height": (100, 300)},
    "cushion": {"long": (300, 700), "short": (300, 700), "height": (100, 350)},
    # ---- Seating ----
    "sofa": {"long": (1200, 3200), "short": (700, 1100), "height": (600, 1100)},
    "couch": {"long": (1200, 3200), "short": (700, 1100), "height": (600, 1100)},
    "sofa bed": {"long": (1500, 2200), "short": (800, 1100), "height": (550, 1100)},
    "loveseat": {"long": (1200, 1800), "short": (700, 1000), "height": (600, 1000)},
    "futon": {"long": (1400, 2100), "short": (700, 1100), "height": (300, 900)},
    "chaise longue": {"long": (1400, 2000), "short": (600, 900), "height": (600, 1000)},
    "recliner": {"long": (700, 1100), "short": (800, 1100), "height": (700, 1100)},
    "chair": {"long": (400, 750), "short": (400, 750), "height": (700, 1200)},
    "armchair": {"long": (600, 1000), "short": (600, 1000), "height": (700, 1100)},
    "rocking chair": {"long": (600, 1100), "short": (500, 700), "height": (700, 1200)},
    "folding chair": {"long": (400, 550), "short": (400, 600), "height": (750, 1000)},
    "deck chair": {"long": (900, 1900), "short": (500, 700), "height": (600, 1100)},
    "highchair": {"long": (400, 600), "short": (400, 600), "height": (800, 1100)},
    "bench": {"long": (900, 2000), "short": (300, 600), "height": (350, 900)},
    "stool": {"long": (300, 600), "short": (300, 600), "height": (350, 800)},
    "step stool": {"long": (300, 550), "short": (300, 500), "height": (250, 600)},
    "footstool": {"long": (300, 600), "short": (300, 500), "height": (250, 500)},
    "ottoman": {"long": (400, 1300), "short": (400, 800), "height": (300, 550)},
    # ---- Tables / desks ----
    "table": {"long": (600, 2400), "short": (500, 1200), "height": (350, 800)},
    # KR/KS ergonomic table & desk height ~700-750mm.
    "dining table": {"long": (900, 2400), "short": (700, 1200), "height": (700, 760)},
    "kitchen table": {"long": (900, 2000), "short": (700, 1100), "height": (700, 760)},
    "coffee table": {"long": (600, 1400), "short": (400, 700), "height": (300, 550)},
    "desk": {"long": (800, 1800), "short": (450, 800), "height": (690, 760)},
    # ---- Storage ----
    "dresser": {"long": (700, 1800), "short": (400, 600), "height": (700, 1300)},
    "drawer": {"long": (400, 1200), "short": (400, 600), "height": (300, 1300)},
    "chest of drawers": {"long": (600, 1300), "short": (400, 600), "height": (600, 1300)},
    # KR 장롱: width in 자 units (1자=303mm), commonly 9-12자 (2727-3636mm);
    # height typically 1900-2100mm, depth ~600mm.
    "wardrobe": {"long": (600, 3700), "short": (550, 700), "height": (1700, 2300)},
    "armoire": {"long": (800, 3700), "short": (500, 700), "height": (1700, 2300)},
    "cabinet": {"long": (400, 1200), "short": (300, 600), "height": (600, 2000)},
    "cupboard": {"long": (500, 1200), "short": (350, 650), "height": (700, 2100)},
    "file cabinet": {"long": (450, 750), "short": (350, 600), "height": (650, 1400)},
    "locker": {"long": (300, 900), "short": (400, 600), "height": (1500, 2000)},
    "bookcase": {"long": (600, 1200), "short": (250, 450), "height": (800, 2400)},
    "bookshelf": {"long": (400, 1200), "short": (250, 450), "height": (700, 2400)},
    "shelf": {"long": (400, 1200), "short": (250, 600), "height": (300, 2400)},
    "piano": {"long": (1300, 1600), "short": (500, 700), "height": (1000, 1400)},
    # ---- Large appliances ----
    "refrigerator": {"long": (550, 1000), "short": (550, 850), "height": (1300, 2000)},
    "automatic washer": {"long": (550, 700), "short": (550, 700), "height": (800, 1000)},
    "washing machine": {"long": (550, 700), "short": (550, 700), "height": (800, 1000)},
    "washer dryer": {"long": (550, 720), "short": (550, 720), "height": (800, 1900)},
    "dishwasher": {"long": (550, 650), "short": (550, 650), "height": (800, 900)},
    "oven": {"long": (550, 750), "short": (550, 700), "height": (550, 950)},
    "stove": {"long": (500, 900), "short": (500, 700), "height": (450, 950)},
    "microwave oven": {"long": (440, 600), "short": (300, 500), "height": (250, 400)},
    "toaster oven": {"long": (350, 550), "short": (300, 450), "height": (250, 400)},
    "water heater": {"long": (400, 650), "short": (400, 650), "height": (900, 1800)},
    "air conditioner": {"long": (700, 1100), "short": (150, 450), "height": (250, 1900)},
    "heater": {"long": (300, 800), "short": (150, 400), "height": (300, 800)},
    "vacuum cleaner": {"long": (250, 500), "short": (250, 450), "height": (250, 1200)},
    "sewing machine": {"long": (350, 600), "short": (200, 350), "height": (250, 400)},
    # ---- Small appliances ----
    "toaster": {"long": (250, 400), "short": (150, 250), "height": (180, 300)},
    "blender": {"long": (150, 250), "short": (150, 250), "height": (300, 500)},
    "coffee maker": {"long": (200, 350), "short": (150, 300), "height": (250, 450)},
    "food processor": {"long": (150, 300), "short": (150, 300), "height": (250, 500)},
    "kettle": {"long": (150, 300), "short": (150, 250), "height": (180, 300)},
    "teakettle": {"long": (150, 300), "short": (150, 250), "height": (180, 300)},
    # ---- Electronics ----
    "tv": {"long": (700, 1800), "short": (40, 150), "height": (400, 800)},
    "television set": {"long": (700, 1800), "short": (40, 150), "height": (400, 800)},
    "computer monitor": {"long": (450, 900), "short": (50, 250), "height": (300, 600)},
    "laptop computer": {"long": (250, 400), "short": (180, 300), "height": (10, 250)},
    "printer": {"long": (350, 550), "short": (300, 500), "height": (150, 400)},
    "speaker (stero equipment)": {"long": (150, 400), "short": (150, 350), "height": (200, 700)},
    "stereo (sound system)": {"long": (200, 450), "short": (250, 400), "height": (80, 300)},
    "subwoofer": {"long": (250, 450), "short": (250, 450), "height": (250, 500)},
    "telephone": {"long": (100, 250), "short": (50, 200), "height": (50, 250)},
    # ---- Lighting / wall decor ----
    "lamp": {"long": (100, 600), "short": (100, 600), "height": (200, 700)},
    "table lamp": {"long": (100, 450), "short": (100, 450), "height": (250, 700)},
    "chandelier": {"long": (300, 1000), "short": (300, 1000), "height": (300, 900)},
    "lantern": {"long": (100, 400), "short": (100, 400), "height": (150, 600)},
    "mirror": {"long": (200, 1200), "short": (10, 60), "height": (300, 2000)},
    "painting": {"long": (300, 2000), "short": (10, 60), "height": (300, 1500)},
    "poster": {"long": (300, 1500), "short": (2, 30), "height": (400, 1500)},
    "clock": {"long": (150, 500), "short": (20, 120), "height": (150, 500)},
    "wall clock": {"long": (200, 600), "short": (20, 80), "height": (200, 600)},
    # ---- Window treatments (thin fabric: small depth) ----
    # A hanging curtain is wide and tall but only fabric-thin in depth, so an
    # inflated BoxerNet depth must be pulled down hard.
    "curtain": {"long": (800, 4000), "short": (10, 120), "height": (900, 2800)},
    # ---- Containers / decor ----
    "vase": {"long": (80, 350), "short": (80, 350), "height": (150, 600)},
    "flowerpot": {"long": (120, 500), "short": (120, 500), "height": (120, 500)},
    # ---- Misc appliance ----
    "fan": {"long": (250, 600), "short": (250, 600), "height": (250, 1400)},
    # ---- Bathroom fixtures ----
    "bathtub": {"long": (1400, 1800), "short": (700, 800), "height": (400, 650)},
    "toilet": {"long": (600, 750), "short": (350, 500), "height": (700, 1000)},
    "sink": {"long": (400, 700), "short": (350, 550), "height": (150, 250)},
    "kitchen sink": {"long": (500, 1000), "short": (400, 600), "height": (150, 300)},
    "washbasin": {"long": (400, 700), "short": (350, 550), "height": (150, 250)},
    "fireplace": {"long": (700, 1600), "short": (300, 600), "height": (700, 1400)},
}

# Genuinely non-cuboidal / amorphous classes: a 3D box is meaningless, so we pass
# them through unchanged rather than impose misleading bounds.
_SKIP = {
    "plant", "tree", "blanket", "bedspread", "quilt",
    "sculpture", "figurine",
}


def _fix(value: float, rng: Range, axis: str, label: str) -> Tuple[float, Optional[str]]:
    lo, hi = rng
    if lo <= value <= hi:
        return value, None
    if value < 0.5 * lo or value > 2.0 * hi:  # severe -> class-typical
        new = (lo + hi) / 2.0
        return new, f"{label}.{axis} {value:.0f}->{new:.0f}mm (severe->typical)"
    new = lo if value < lo else hi  # mild -> clamp to nearest bound
    return new, f"{label}.{axis} {value:.0f}->{new:.0f}mm (clamp)"


def sanitize_dims(
    label: str,
    width_mm: float,
    depth_mm: float,
    height_mm: float,
) -> Tuple[float, float, float, List[str]]:
    """Clamp (w, d, h) to the class's plausible range. Unknown class -> unchanged."""
    key = label.lower().strip()
    bounds = _BOUNDS.get(key)
    if bounds is None:
        if key not in _SKIP:
            logger.debug("no dimension bounds for class '%s' (pass-through)", key)
        return width_mm, depth_mm, height_mm, []

    corrections: List[str] = []
    h, ch = _fix(height_mm, bounds["height"], "height", label)
    if ch:
        corrections.append(ch)

    # Horizontal footprint: larger dim -> `long`, smaller -> `short`, then map back.
    width_is_long = width_mm >= depth_mm
    long_v, short_v = (width_mm, depth_mm) if width_is_long else (depth_mm, width_mm)
    long_f, cl = _fix(long_v, bounds["long"], "long", label)
    short_f, cs = _fix(short_v, bounds["short"], "short", label)
    if cl:
        corrections.append(cl)
    if cs:
        corrections.append(cs)
    w, d = (long_f, short_f) if width_is_long else (short_f, long_f)

    return w, d, h, corrections
