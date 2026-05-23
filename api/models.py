"""Pydantic request/response models. No ply_url / GCS fields."""

from typing import List

from pydantic import BaseModel, Field


class ImageUrlItem(BaseModel):
    id: int
    url: str


class AnalyzeFurnitureRequest(BaseModel):
    """Multi-image analysis (async callback). Mirrors Isajjim-AI TDD §4.1.

    동시성 제어는 GPU 풀이 담당하므로 별도 `max_concurrent` 필드가 없다.
    """

    estimate_id: int
    image_urls: List[ImageUrlItem] = Field(..., min_length=1, max_length=20)


class AnalyzeFurnitureSingleRequest(BaseModel):
    image_url: str


class AnalyzeFurnitureBase64Request(BaseModel):
    image: str
    enable_3d: bool = True
