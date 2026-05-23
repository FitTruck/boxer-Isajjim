"""Stage 1: Fetch images from Firebase Storage / arbitrary HTTPS URLs."""

import io
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ImageFetcher:
    """Download images from URLs (async + sync)."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    async def fetch_async(self, url: str) -> Optional[Image.Image]:
        if url.startswith("file://") or (url.startswith("/") and not url.startswith("//")):
            return self._load_local(url.removeprefix("file://"))

        if not HAS_AIOHTTP:
            return self.fetch_sync(url)

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

    def fetch_sync(self, url: str) -> Optional[Image.Image]:
        if not HAS_REQUESTS:
            logger.error("requests not installed")
            return None
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"fetch_sync failed for {url}: {e}")
            return None

    @staticmethod
    def _load_local(path: str) -> Optional[Image.Image]:
        import os

        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Local read failed {path}: {e}")
            return None

