"""API package — re-exports the FastAPI app for uvicorn."""

from api.app import app

__all__ = ["app"]
