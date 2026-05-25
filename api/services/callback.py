"""Send analysis result to backend callback URL."""

import logging
from typing import Any, Dict, Optional

import aiohttp

from api.config import (
    CALLBACK_RETRY_COUNT,
    CALLBACK_TIMEOUT_SECONDS,
    CALLBACK_URL_TEMPLATE,
    X_INTERNAL_TOKEN,
)

logger = logging.getLogger(__name__)


async def send_callback(
    estimate_id: int,
    result_data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> bool:
    """POST result/error JSON to `{CALLBACK_URL_TEMPLATE}` with retries.

    Returns:
        True on 2xx response, False otherwise (logs the failure).
    """
    url = CALLBACK_URL_TEMPLATE.replace("{estimateId}", str(estimate_id))
    payload: Dict[str, Any] = {"error": error} if error else (result_data or {"error": "Unknown error"})

    if not X_INTERNAL_TOKEN:
        logger.warning("AUTH_TOKEN env var is empty; backend will reject the callback")

    headers = {"X-INTERNAL-TOKEN": X_INTERNAL_TOKEN}
    timeout = aiohttp.ClientTimeout(total=CALLBACK_TIMEOUT_SECONDS)
    for attempt in range(CALLBACK_RETRY_COUNT + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"[Callback] OK estimate_id={estimate_id} status={response.status}")
                        return True
                    body = await response.text()
                    logger.warning(
                        f"[Callback] FAIL estimate_id={estimate_id} status={response.status} body={body[:200]}"
                    )
        except aiohttp.ClientError as e:
            logger.warning(f"[Callback] network error attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"[Callback] unexpected error: {e}")
    logger.error(f"[Callback] all retries failed estimate_id={estimate_id}")
    return False
