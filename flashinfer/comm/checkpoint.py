"""Checkpoint lifecycle helpers for graph-visible symmetric CUDA memory."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, Optional

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


class SymmetricMemoryState(Enum):
    """Attachment state of a checkpointable symmetric allocation."""

    ATTACHED = auto()
    DETACHING = auto()
    DETACHED = auto()
    RESTORING = auto()


def _check_driver(result: Any, operation: str, driver: Any) -> Any:
    error, *values = result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {error}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class SymmetricMemoryCheckpoint:
    """Renew symmetric-memory handles while preserving CUDA virtual addresses.

    CUDA graphs capture virtual addresses, not allocation handles. Detaching
    releases the process- and device-specific backing while retaining the
    address reservations. Restoring creates new shared allocations and maps
    them at those original addresses.
    """

    def __init__(
        self,
        memory: Any,
        *,
        enable_multicast: Optional[bool] = None,
        _driver_factory: Callable[[], Any] = lambda: cuda,
        _memory_factory: Optional[Callable[..., Any]] = None,
    ):
        self.memory = memory
        self.enable_multicast = (
            bool(memory.mc_ptr) if enable_multicast is None else enable_multicast
        )
        self._driver_factory = _driver_factory
        self._memory_factory = _memory_factory
        self.state = SymmetricMemoryState.ATTACHED
        self._unmapped_unicast: set[int] = set()
        self._unmapped_multicast = False

    @property
    def attached(self) -> bool:
        return self.state is SymmetricMemoryState.ATTACHED

    def detach(self) -> None:
        """Unmap and release backing allocations, retaining virtual addresses."""
        if self.state is SymmetricMemoryState.DETACHED:
            return
        if self.state is SymmetricMemoryState.RESTORING:
            raise RuntimeError("cannot detach symmetric memory while restoring")
        if self.state is SymmetricMemoryState.ATTACHED:
            self.state = SymmetricMemoryState.DETACHING

        driver = self._driver_factory()
        memory = self.memory
        try:
            for peer, handle in enumerate(memory.uc_handles):
                if not handle:
                    continue
                if peer not in self._unmapped_unicast:
                    _check_driver(
                        driver.cuMemUnmap(memory.uc_ptrs[peer], memory.allocation_size),
                        "cuMemUnmap(unicast)",
                        driver,
                    )
                    self._unmapped_unicast.add(peer)
                _check_driver(
                    driver.cuMemRelease(handle),
                    "cuMemRelease(unicast)",
                    driver,
                )
                memory.uc_handles[peer] = 0

            if memory.mc_handle:
                if not self._unmapped_multicast:
                    _check_driver(
                        driver.cuMemUnmap(memory.mc_ptr, memory.allocation_size),
                        "cuMemUnmap(multicast)",
                        driver,
                    )
                    self._unmapped_multicast = True
                _check_driver(
                    driver.cuMemRelease(memory.mc_handle),
                    "cuMemRelease(multicast)",
                    driver,
                )
                memory.mc_handle = 0

            exchanger = getattr(memory, "_exchanger", None)
            if exchanger is not None:
                exchanger.close()
                del memory._exchanger
            memory.comm_backend = None
        except Exception:
            self.state = SymmetricMemoryState.DETACHING
            raise
        self.state = SymmetricMemoryState.DETACHED

    def restore(self, comm_backend: Any) -> None:
        """Create fresh backing and map it at the graph-visible addresses."""
        if self.state is SymmetricMemoryState.ATTACHED:
            return
        if self.state is SymmetricMemoryState.DETACHING:
            raise RuntimeError("symmetric memory detach did not complete")
        if self.state is SymmetricMemoryState.RESTORING:
            raise RuntimeError("symmetric memory restore is already in progress")
        if comm_backend.Get_rank() != self.memory.group_rank:
            raise ValueError("communication backend rank changed across restore")
        if comm_backend.Get_size() != self.memory.group_size:
            raise ValueError("communication backend size changed across restore")

        self.state = SymmetricMemoryState.RESTORING
        driver = self._driver_factory()
        fresh = None
        mapped_unicast: list[int] = []
        mapped_multicast = False
        try:
            fresh = self._create_fresh(comm_backend)
            self._release_fresh_addresses(fresh, driver)

            for peer, handle in enumerate(fresh.uc_handles):
                _check_driver(
                    driver.cuMemMap(
                        self.memory.uc_ptrs[peer],
                        self.memory.allocation_size,
                        0,
                        handle,
                        0,
                    ),
                    "cuMemMap(restored unicast)",
                    driver,
                )
                mapped_unicast.append(peer)
            _check_driver(
                driver.cuMemSetAccess(
                    self.memory.uc_base_ptr,
                    self.memory.total_uc_size,
                    [self.memory._get_mem_access_desc()],
                    1,
                ),
                "cuMemSetAccess(restored unicast)",
                driver,
            )

            if fresh.mc_handle:
                _check_driver(
                    driver.cuMemMap(
                        self.memory.mc_ptr,
                        self.memory.allocation_size,
                        0,
                        fresh.mc_handle,
                        0,
                    ),
                    "cuMemMap(restored multicast)",
                    driver,
                )
                mapped_multicast = True
                _check_driver(
                    driver.cuMemSetAccess(
                        self.memory.mc_ptr,
                        self.memory.allocation_size,
                        [self.memory._get_mem_access_desc()],
                        1,
                    ),
                    "cuMemSetAccess(restored multicast)",
                    driver,
                )

            _check_driver(
                driver.cuMemsetD8(
                    self.memory.uc_ptrs[self.memory.group_rank],
                    0,
                    self.memory.allocation_size,
                ),
                "cuMemsetD8(restored local allocation)",
                driver,
            )
            self._adopt_fresh(fresh, comm_backend)
        except Exception:
            if mapped_multicast:
                driver.cuMemUnmap(self.memory.mc_ptr, self.memory.allocation_size)
            for peer in reversed(mapped_unicast):
                driver.cuMemUnmap(
                    self.memory.uc_ptrs[peer], self.memory.allocation_size
                )
            if fresh is not None:
                self._cleanup_fresh(fresh, driver)
            self.state = SymmetricMemoryState.DETACHED
            raise

        self._unmapped_unicast.clear()
        self._unmapped_multicast = False
        self.state = SymmetricMemoryState.ATTACHED

    def _create_fresh(self, comm_backend: Any) -> Any:
        factory = self._memory_factory
        if factory is None:
            from .mnnvl import SymmDeviceMemory

            factory = SymmDeviceMemory
        fresh = factory(
            buf_size=self.memory.buf_size,
            group_size=self.memory.group_size,
            group_rank=self.memory.group_rank,
            device_idx=self.memory.device_idx,
            comm_backend_for_handle_transfer=comm_backend,
            enable_multicast=self.enable_multicast,
            allocate_signal_pads=False,
        )
        if fresh.allocation_size != self.memory.allocation_size:
            self._cleanup_fresh(fresh, self._driver_factory())
            raise RuntimeError("symmetric allocation size changed across restore")
        return fresh

    @staticmethod
    def _release_fresh_addresses(fresh: Any, driver: Any) -> None:
        for peer, peer_ptr in enumerate(fresh.uc_ptrs):
            _check_driver(
                driver.cuMemUnmap(peer_ptr, fresh.allocation_size),
                "cuMemUnmap(fresh unicast)",
                driver,
            )
            fresh.uc_ptrs[peer] = 0
        _check_driver(
            driver.cuMemAddressFree(fresh.uc_base_ptr, fresh.total_uc_size),
            "cuMemAddressFree(fresh unicast)",
            driver,
        )
        fresh.uc_base_ptr = 0

        if fresh.mc_handle:
            _check_driver(
                driver.cuMemUnmap(fresh.mc_ptr, fresh.allocation_size),
                "cuMemUnmap(fresh multicast)",
                driver,
            )
            _check_driver(
                driver.cuMemAddressFree(fresh.mc_ptr, fresh.allocation_size),
                "cuMemAddressFree(fresh multicast)",
                driver,
            )
            fresh.mc_ptr = 0

        if fresh.uc_ptrs_dev:
            _check_driver(
                driver.cuMemFree(fresh.uc_ptrs_dev),
                "cuMemFree(fresh pointer table)",
                driver,
            )
            fresh.uc_ptrs_dev = 0

    def _adopt_fresh(self, fresh: Any, comm_backend: Any) -> None:
        self.memory.uc_handles = fresh.uc_handles
        self.memory.mc_handle = fresh.mc_handle
        self.memory.comm_backend = comm_backend
        self.memory._exchanger = fresh._exchanger
        self._disarm_fresh(fresh)

    @classmethod
    def _cleanup_fresh(cls, fresh: Any, driver: Any) -> None:
        for peer, peer_ptr in enumerate(fresh.uc_ptrs):
            if peer_ptr:
                driver.cuMemUnmap(peer_ptr, fresh.allocation_size)
                fresh.uc_ptrs[peer] = 0
        if fresh.uc_base_ptr:
            driver.cuMemAddressFree(fresh.uc_base_ptr, fresh.total_uc_size)
            fresh.uc_base_ptr = 0
        if fresh.mc_ptr:
            driver.cuMemUnmap(fresh.mc_ptr, fresh.allocation_size)
            driver.cuMemAddressFree(fresh.mc_ptr, fresh.allocation_size)
            fresh.mc_ptr = 0
        if fresh.uc_ptrs_dev:
            driver.cuMemFree(fresh.uc_ptrs_dev)
            fresh.uc_ptrs_dev = 0
        for peer, handle in enumerate(fresh.uc_handles):
            if handle:
                driver.cuMemRelease(handle)
                fresh.uc_handles[peer] = 0
        if fresh.mc_handle:
            driver.cuMemRelease(fresh.mc_handle)
            fresh.mc_handle = 0
        exchanger = getattr(fresh, "_exchanger", None)
        if exchanger is not None:
            exchanger.close()
            del fresh._exchanger
        cls._disarm_fresh(fresh)

    @staticmethod
    def _disarm_fresh(fresh: Any) -> None:
        fresh.uc_handles = []
        fresh.uc_ptrs = []
        fresh.uc_base_ptr = 0
        fresh.uc_ptrs_dev = 0
        fresh.mc_handle = 0
        fresh.mc_ptr = 0
        if hasattr(fresh, "_exchanger"):
            del fresh._exchanger
