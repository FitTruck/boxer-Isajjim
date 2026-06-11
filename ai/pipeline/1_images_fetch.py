"""Stage 1: Fetch images from Firebase Storage / arbitrary HTTPS URLs."""

import io
import logging
import os
from typing import Optional

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)


class ImageFetcher:
    """Download images from URLs (async); ``file://`` / absolute paths load locally."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    async def fetch_async(self, url: str) -> Optional[Image.Image]:
        if url.startswith("file://") or (url.startswith("/") and not url.startswith("//")):
            return self._load_local(url.removeprefix("file://"))

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
                    data = await response.read()
                    return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            logger.warning(f"fetch_async failed for {url}: {e}")
            return None

    @staticmethod
    def _load_local(path: str) -> Optional[Image.Image]:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Local read failed {path}: {e}")
            return None
