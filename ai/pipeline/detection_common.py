"""Shared output contract for the 2D detector backends (OWLv2 / SAM3).

Both detectors return the same ``{"boxes", "scores", "labels"}`` dict and use
the same chunk-merge tail (concatenate -> class-agnostic NMS -> id->name map).
Owning both here keeps the two backends from drifting apart.
"""

from typing import Dict, List

import numpy as np
import torch
from torchvision.ops import nms


def empty_detections() -> Dict[str, np.ndarray]:
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": [],
    }


def merge_chunks_nms(
    all_boxes: List[np.ndarray],
    all_scores: List[np.ndarray],
    all_label_ids: List[np.ndarray],
    names: List[str],
    iou: float,
) -> Dict[str, np.ndarray]:
    """Merge per-chunk detections: concat -> class-agnostic NMS -> name lookup.

    Chunks (text-prompt batches / per-concept passes) may double-detect the
    same object, hence class-agnostic suppression across the whole set.
    """
    boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    label_ids = np.concatenate(all_label_ids, axis=0).astype(int)

    keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).tolist()
    boxes, scores, label_ids = boxes[keep], scores[keep], label_ids[keep]
    return {
        "boxes": boxes,
        "scores": scores,
        "labels": [names[i] for i in label_ids],
    }
