"""Unit tests for GPUPoolManager (semaphore + multi-device dispatch)."""

import asyncio
import time

import pytest

from ai.gpu import GPUPoolManager


class TestRoundRobinDispatch:
    @pytest.mark.asyncio
    async def test_acquire_returns_round_robin(self):
        pool = GPUPoolManager(["cuda:0", "cuda:1", "cuda:2"])
        seen = []
        for i in range(3):
            d = await pool.acquire(task_id=f"t{i}")
            seen.append(d)
        assert sorted(seen) == ["cuda:0", "cuda:1", "cuda:2"]
        # release all
        for d in seen:
            await pool.release(d)

    @pytest.mark.asyncio
    async def test_release_is_idempotent(self):
        pool = GPUPoolManager(["mps"])
        d = await pool.acquire("t")
        await pool.release(d)
        await pool.release(d)  # second release should not over-credit the semaphore
        # we can still acquire exactly once before blocking
        d2 = await pool.acquire("t2")
        assert d2 == "mps"
        # next acquire should block — wait briefly and confirm timeout
        with pytest.raises(RuntimeError, match="timeout"):
            await pool.acquire("t3", wait_timeout=0.2)


class TestSemaphoreNoPolling:
    @pytest.mark.asyncio
    async def test_acquire_wakes_immediately_on_release(self):
        """Release() should wake a waiting acquire() without polling delay."""
        pool = GPUPoolManager(["cpu"])
        d = await pool.acquire("first")

        async def releaser():
            await asyncio.sleep(0.05)  # hold briefly
            await pool.release(d)

        async def waiter():
            return await pool.acquire("second")

        start = time.perf_counter()
        results = await asyncio.gather(releaser(), waiter())
        elapsed = time.perf_counter() - start

        assert results[1] == "cpu"
        # Old polling impl would round up to >= 500ms; semaphore should be ~50ms.
        assert elapsed < 0.3, f"acquire took {elapsed:.3f}s — likely still polling"


class TestStatusReporting:
    def test_status_reports_devices(self):
        pool = GPUPoolManager(["cuda:0", "mps"])
        status = pool.get_status()
        assert status["total_devices"] == 2
        assert status["available_devices"] == 2
        assert set(status["devices"].keys()) == {"cuda:0", "mps"}

    @pytest.mark.asyncio
    async def test_status_reflects_busy_slot(self):
        pool = GPUPoolManager(["cuda:0", "cuda:1"])
        d = await pool.acquire("busy")
        status = pool.get_status()
        assert status["available_devices"] == 1
        assert status["devices"][d]["available"] is False
        assert status["devices"][d]["task_id"] == "busy"
        await pool.release(d)
