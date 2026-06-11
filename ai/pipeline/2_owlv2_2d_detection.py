"""Stage 2: OWLv2 open-vocabulary 2D detection.

Class vocabulary is loaded from `ai/pipeline/lvisplus_classes.csv` (curated
LVIS+ subset). OWLv2 attends to all text prompts at once; for large
vocabularies we chunk the queries and merge with class-agnostic NMS.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from ai.config import Config
from ai.pipeline.detection_common import empty_detections, merge_chunks_nms

logger = logging.getLogger(__name__)

_DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "lvisplus_classes.csv")


def _load_lvis_classes(csv_path: str) -> List[str]:
    """Read one class name per row from the LVIS+ CSV (drop blanks)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


class Owlv2Detector:
    """OWLv2 open-vocabulary detector (replaces YOLOE)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        classes_csv: Optional[str] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
        chunk_size: Optional[int] = None,
        nms_iou: float = 0.5,
    ):
        self.model_name = model_name or Config.OWLV2_MODEL
        self.confidence = confidence if confidence is not None else Config.OWLV2_CONFIDENCE
        self.device = device or Config.get_default_device()
        self.chunk_size = chunk_size or Config.OWLV2_CHUNK_SIZE
        self.nms_iou = nms_iou
        self.classes_csv = classes_csv or Config.OWLV2_CLASSES_CSV or _DEFAULT_CSV

        self.classes: List[str] = _load_lvis_classes(self.classes_csv)
        # CSV stores LVIS-style names with underscores; surface to callers as
        # space-separated for readability.
        self.display_labels: List[str] = [c.replace("_", " ") for c in self.classes]
        # Text prompt fed to OWLv2 — natural-language template helps recall.
        self.prompts: List[str] = [f"a photo of a {lbl}" for lbl in self.display_labels]

        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info(
            f"Loading OWLv2 on {self.device}: {self.model_name} ({len(self.classes)} classes)"
        )
        self.processor = Owlv2Processor.from_pretrained(self.model_name)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device).eval()
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def detect(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """Run OWLv2 with chunked text queries, NMS-merge across chunks."""
        if self.model is None:
            return empty_detections()

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_label_ids: List[np.ndarray] = []  # global LVIS class id

        target = torch.tensor([(image.height, image.width)], device=self.device)
        for start in range(0, len(self.prompts), self.chunk_size):
            stop = start + self.chunk_size
            prompts_chunk = self.prompts[start:stop]
            inputs = self.processor(
                text=[prompts_chunk], images=image, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            post = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target, threshold=self.confidence
            )[0]
            if len(post["boxes"]) == 0:
                continue
            all_boxes.append(post["boxes"].detach().cpu().numpy())
            all_scores.append(post["scores"].detach().cpu().numpy())
            # `labels` are indices into the chunk; shift to global id.
            local_ids = post["labels"].detach().cpu().numpy()
            all_label_ids.append(local_ids + start)

        if not all_boxes:
            return empty_detections()
        return merge_chunks_nms(
            all_boxes, all_scores, all_label_ids, self.display_labels, self.nms_iou
        )

