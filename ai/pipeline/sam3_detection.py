"""Stage 2 (alternative): SAM3 open-vocabulary 2D detection.

Drop-in alternative to `Owlv2Detector` — `detect()` returns the same
``{"boxes", "scores", "classes", "labels"}`` dict so BoxerNet lifting is
unchanged.

Mechanics differ from OWLv2: SAM3 (Promptable Concept Segmentation) prompts
ONE concept (noun phrase) at a time and returns every matching instance. To
avoid re-running the heavy vision backbone per concept, we compute the image
vision features once and reuse them across concepts (one heavy backbone pass +
N light decoder passes), per the facebook/sam3 transformers docs. Cost still
scales with the number of concepts, so keep the taxonomy small.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from ai.config import Config

logger = logging.getLogger(__name__)

# Compact furniture taxonomy. Keep short — SAM3 cost grows with concept count.
# Override via SAM3_CONCEPTS_CSV (one per line) or SAM3_CONCEPTS env (comma-sep).
_DEFAULT_CONCEPTS: List[str] = [
    "sofa",
    "armchair",
    "chair",
    "stool",
    "dining table",
    "coffee table",
    "desk",
    "bed",
    "wardrobe",
    "dresser",
    "chest of drawers",
    "bookshelf",
    "shelf",
    "cabinet",
    "tv",
    "refrigerator",
    "washing machine",
    "air conditioner",
    "lamp",
]


def _load_concepts(csv_path: Optional[str]) -> List[str]:
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    env = os.environ.get("SAM3_CONCEPTS")
    if env:
        return [c.strip() for c in env.split(",") if c.strip()]
    return list(_DEFAULT_CONCEPTS)


class Sam3Detector:
    """SAM3 open-vocabulary detector with the Owlv2Detector output contract."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        concepts_csv: Optional[str] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
        nms_iou: float = 0.5,
    ):
        self.model_name = model_name or Config.SAM3_MODEL
        self.confidence = confidence if confidence is not None else Config.SAM3_CONFIDENCE
        self.device = device or Config.get_default_device()
        self.nms_iou = nms_iou
        self.concepts: List[str] = _load_concepts(concepts_csv or Config.SAM3_CONCEPTS_CSV)

        self.processor = None
        self.model = None
        self._reuse_vision = True  # disabled if get_vision_features path errors once
        self._load_model()

    def _load_model(self) -> None:
        from transformers import Sam3Model, Sam3Processor

        logger.info(
            f"Loading SAM3 on {self.device}: {self.model_name} ({len(self.concepts)} concepts)"
        )
        self.processor = Sam3Processor.from_pretrained(self.model_name)
        self.model = Sam3Model.from_pretrained(self.model_name).to(self.device).eval()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def detect(self, image: Image.Image) -> Dict[str, np.ndarray]:
        if self.model is None:
            return self._empty()

        target_sizes = [(image.height, image.width)]

        # Heavy vision backbone once; reuse across concept prompts when supported.
        vision_embeds = None
        if self._reuse_vision:
            try:
                img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                vision_embeds = self.model.get_vision_features(
                    pixel_values=img_inputs.pixel_values
                )
            except Exception as e:
                logger.warning(
                    f"SAM3 vision-feature reuse unavailable ({e}); using per-concept full forward"
                )
                self._reuse_vision = False

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_cids: List[np.ndarray] = []
        for cid, concept in enumerate(self.concepts):
            try:
                if vision_embeds is not None:
                    text_inputs = self.processor(text=concept, return_tensors="pt").to(self.device)
                    outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
                else:
                    inputs = self.processor(
                        images=image, text=concept, return_tensors="pt"
                    ).to(self.device)
                    outputs = self.model(**inputs)
                res = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.confidence,
                    mask_threshold=0.5,
                    target_sizes=target_sizes,
                )[0]
            except Exception as e:
                logger.warning(f"SAM3 concept '{concept}' failed: {e}")
                continue

            boxes = res.get("boxes")
            scores = res.get("scores")
            if boxes is None or len(boxes) == 0:
                continue
            boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
            scores_np = scores.detach().cpu().numpy().astype(np.float32)
            all_boxes.append(boxes_np)
            all_scores.append(scores_np)
            all_cids.append(np.full(len(boxes_np), cid, dtype=int))

        if not all_boxes:
            return self._empty()

        boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        scores = np.concatenate(all_scores, axis=0).astype(np.float32)
        cids = np.concatenate(all_cids, axis=0).astype(int)

        # Class-agnostic NMS — different concepts may detect the same object.
        keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), self.nms_iou).tolist()
        boxes = boxes[keep]
        scores = scores[keep]
        cids = cids[keep]
        labels = [self.concepts[i] for i in cids]

        return {"boxes": boxes, "scores": scores, "classes": cids, "labels": labels}

    @staticmethod
    def _empty() -> Dict[str, np.ndarray]:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=int),
            "labels": [],
        }
