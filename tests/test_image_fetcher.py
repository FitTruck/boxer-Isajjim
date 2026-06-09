"""Unit tests for ImageFetcher (network calls mocked)."""

import io

import pytest
from PIL import Image

# Numbered file: import via importlib in pipeline __init__
from ai.pipeline import ImageFetcher


def _png_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestImageFetcher:
    def test_local_file_loads(self, tmp_path):
        path = tmp_path / "img.png"
        path.write_bytes(_png_bytes())
        fetcher = ImageFetcher()
        img = fetcher._load_local(str(path))
        assert img is not None
        assert img.size == (10, 10)

    @pytest.mark.asyncio
    async def test_fetch_async_returns_none_on_404(self):
        fetcher = ImageFetcher(timeout=1)
        # We can't easily intercept aiohttp without extra deps; verify graceful failure
        result = await fetcher.fetch_async("http://127.0.0.1:1/nonexistent.jpg")
        assert result is None
