"""Multi-device pool with semaphore-based event scheduling.

서버 시작 시 디바이스당 1개의 FurniturePipeline (OWLv2 + Depth + Boxer)을 미리 로드하고,
요청마다 사용 가능한 디바이스 슬롯을 round-robin으로 분배한다.

스케줄링은 `asyncio.Semaphore` 기반:
- 슬롯 수만큼의 토큰을 보유한 세마포어 1개
- acquire() = 토큰 1개 획득 → 라운드로빈으로 빈 슬롯 찾기 → 마킹 후 반환
- release() = 슬롯 해제 → 토큰 반환 → 대기 중이던 코루틴 즉시 깨어남

폴링 sleep 없이 대기 코루틴이 즉시 dispatch되므로 큐잉 latency가 사실상 0.
디바이스 문자열은 "cuda:N" / "mps" / "cpu" 형태 — 단일 풀이 멀티 백엔드 지원.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _Slot:
    device: str  # "cuda:0" / "mps" / "cpu"
    available: bool = True
    task_id: Optional[str] = None
    pipeline: object = None


class GPUPoolManager:
    """Multi-device resource pool with semaphore scheduling."""

    def __init__(self, devices: List[str]):
        self.devices = list(devices) if devices else ["cpu"]
        self._slots: Dict[str, _Slot] = {d: _Slot(device=d) for d in self.devices}
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(len(self.devices))
        self._idx = 0  # round-robin cursor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize_pipelines(
        self,
        factory: Callable[[str], object],
        skip_on_error: bool = True,
    ) -> int:
        """Run `factory(device)` on each device at startup."""
        ok = 0
        for device in self.devices:
            try:
                logger.info(f"[{device}] initializing pipeline...")
                self._slots[device].pipeline = factory(device)
                ok += 1
            except Exception as e:
                if not skip_on_error:
                    raise
                logger.error(f"[{device}] pipeline init failed: {e}")
        logger.info(f"GPU pool ready: {ok}/{len(self.devices)} pipelines initialized")
        return ok

    def has_pipeline(self, device: str) -> bool:
        return device in self._slots and self._slots[device].pipeline is not None

    # ------------------------------------------------------------------
    # Acquire / release  (semaphore-based, no polling)
    # ------------------------------------------------------------------
    async def acquire(self, task_id: str, wait_timeout: float = 300.0) -> str:
        """Wait for a free device slot. Returns device string."""
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"GPU pool acquire timeout (task={task_id})") from None

        try:
            async with self._lock:
                for _ in range(len(self.devices)):
                    device = self.devices[self._idx % len(self.devices)]
                    self._idx += 1
                    slot = self._slots[device]
                    if slot.available:
                        slot.available = False
                        slot.task_id = task_id
                        return device
        except Exception:
            self._sem.release()
            raise

        # Semaphore promised a free slot but none found — should be impossible.
        self._sem.release()
        raise RuntimeError("GPU pool inconsistency: no free slot despite semaphore token")

    async def release(self, device: str) -> None:
        """Release a slot. Idempotent — safe to call twice."""
        notify = False
        async with self._lock:
            slot = self._slots.get(device)
            if slot is not None and not slot.available:
                slot.available = True
                slot.task_id = None
                notify = True
        if notify:
            self._sem.release()  # wakes one waiter immediately

    @asynccontextmanager
    async def pipeline_context(self, task_id: str):
        device = await self.acquire(task_id)
        try:
            yield device, self._slots[device].pipeline
        finally:
            await self.release(device)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        try:
            import torch

            mem_for_cuda = torch.cuda.is_available()
        except Exception:
            mem_for_cuda = False

        devices: Dict[str, Dict] = {}
        for device, slot in self._slots.items():
            mem_mb = 0
            if mem_for_cuda and device.startswith("cuda:"):
                try:
                    idx = int(device.split(":")[1])
                    mem_mb = int(torch.cuda.memory_allocated(idx) / (1024 * 1024))
                except Exception:
                    pass
            devices[device] = {
                "available": slot.available,
                "task_id": slot.task_id,
                "memory_used_mb": mem_mb,
                "has_pipeline": slot.pipeline is not None,
            }

        return {
            "total_devices": len(self.devices),
            "available_devices": sum(1 for s in self._slots.values() if s.available),
            "pipelines_initialized": sum(
                1 for s in self._slots.values() if s.pipeline is not None
            ),
            "devices": devices,
        }


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------
_pool: Optional[GPUPoolManager] = None


def initialize_gpu_pool(devices: List[str]) -> GPUPoolManager:
    global _pool
    _pool = GPUPoolManager(devices)
    return _pool


def get_gpu_pool() -> GPUPoolManager:
    if _pool is None:
        raise RuntimeError("GPU pool not initialized")
    return _pool


async def shutdown_gpu_pool() -> None:
    global _pool
    _pool = None
