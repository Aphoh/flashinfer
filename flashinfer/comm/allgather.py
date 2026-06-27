"""CUDA-graph-safe all-gather over peer-mapped symmetric memory."""

from __future__ import annotations

from functools import cache
from typing import Optional

import torch

from flashinfer.jit.comm import gen_symmetric_allgather_module

from .checkpoint import SymmetricMemoryCheckpoint
from .mnnvl import CommBackend, SymmDeviceMemory


@cache
def _get_module():
    return gen_symmetric_allgather_module().build_and_load()


class SymmetricAllGatherWorkspace:
    """Persistent peer-mapped workspace for a fixed-size all-gather group."""

    _DTYPE_CODES = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
    }

    def __init__(
        self,
        max_elems: int,
        world_size: int,
        rank: int,
        comm_backend: CommBackend,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if max_elems <= 0:
            raise ValueError("max_elems must be positive")
        if world_size <= 0 or world_size > 16:
            raise ValueError("world_size must be in [1, 16]")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        if dtype not in self._DTYPE_CODES:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if comm_backend.Get_rank() != rank:
            raise ValueError("comm_backend rank does not match rank")
        if comm_backend.Get_size() != world_size:
            raise ValueError("comm_backend size does not match world_size")

        self.max_elems = max_elems
        self.world_size = world_size
        self.rank = rank
        self.dtype = dtype
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._module = _get_module()
        self._dtype_code = self._DTYPE_CODES[dtype]
        self._element_size = torch.empty((), dtype=dtype).element_size()
        workspace_bytes = int(
            self._module.get_workspace_bytes(max_elems, world_size, self._element_size)
        )
        self._memory = SymmDeviceMemory(
            buf_size=workspace_bytes,
            group_size=world_size,
            group_rank=rank,
            device_idx=self.device.index,
            comm_backend_for_handle_transfer=comm_backend,
            enable_multicast=False,
            allocate_signal_pads=False,
        )
        self._checkpoint = SymmetricMemoryCheckpoint(
            self._memory, enable_multicast=False
        )
        self._peer_ptrs = tuple(int(ptr) for ptr in self._memory.get_buffer_ptrs_host())
        self._default_ticket = torch.empty(2, dtype=torch.uint64, device=self.device)
        self._capture_tickets: list[torch.Tensor] = []
        self._closed = False
        self._initialize_protocol()
        comm_backend.barrier()

    @property
    def checkpoint_detached(self) -> bool:
        return not self._checkpoint.attached

    @property
    def peer_addresses(self) -> tuple[int, ...]:
        return self._peer_ptrs

    def all_gather(
        self,
        input: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather flattened rank inputs into rank-major contiguous output."""
        self._check_attached()
        self._validate_input(input)
        if output is None:
            output = torch.empty(
                (input.shape[0] * self.world_size, *input.shape[1:]),
                dtype=input.dtype,
                device=input.device,
            )
        self._validate_output(input, output)

        if torch.cuda.is_current_stream_capturing():
            ticket = torch.empty(2, dtype=torch.uint64, device=self.device)
            self._capture_tickets.append(ticket)
        else:
            ticket = self._default_ticket

        stream = torch.cuda.current_stream(self.device).cuda_stream
        self._module.reserve_sequences(self._peer_ptrs[self.rank], ticket, 1, stream)
        vector_bytes = 16
        threads = 256
        blocks = min(
            (input.numel() * self._element_size + vector_bytes * threads - 1)
            // (vector_bytes * threads),
            torch.cuda.get_device_properties(self.device).multi_processor_count,
        )
        self._module.launch_allgather(
            self._peer_ptrs[self.rank],
            list(self._peer_ptrs),
            input,
            output,
            input.numel(),
            self.max_elems,
            self.world_size,
            self.rank,
            max(blocks, 1),
            threads,
            self._dtype_code,
            ticket,
            stream,
        )
        return output

    def prepare_checkpoint(self) -> None:
        """Release device-specific backing while retaining graph addresses."""
        self._check_open()
        if self.checkpoint_detached:
            return
        torch.cuda.synchronize(self.device)
        self._checkpoint.detach()

    def restore_after_checkpoint(self, comm_backend: CommBackend) -> None:
        """Renew peer handles and reset protocol state after process restore."""
        self._check_open()
        if not self.checkpoint_detached:
            return
        self._checkpoint.restore(comm_backend)
        restored_ptrs = tuple(int(ptr) for ptr in self._memory.get_buffer_ptrs_host())
        if restored_ptrs != self._peer_ptrs:
            raise RuntimeError("graph-visible all-gather pointers changed on restore")
        self._initialize_protocol()
        comm_backend.barrier()

    def status(self) -> list[int]:
        self._check_attached()
        return [
            int(value)
            for value in self._module.get_status(
                self._peer_ptrs[self.rank],
                self.max_elems,
                self.world_size,
                self._element_size,
            )
        ]

    def destroy(self) -> None:
        if self._closed:
            return
        self._capture_tickets.clear()
        self._default_ticket = None
        self._checkpoint = None
        self._memory = None
        self._closed = True

    close = destroy

    def _initialize_protocol(self) -> None:
        self._module.initialize_workspace(
            self._peer_ptrs[self.rank],
            self.max_elems,
            self.world_size,
            self._element_size,
            torch.cuda.current_stream(self.device).cuda_stream,
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("all-gather workspace is closed")

    def _check_attached(self) -> None:
        self._check_open()
        if self.checkpoint_detached:
            raise RuntimeError("all-gather workspace is detached")

    def _validate_input(self, input: torch.Tensor) -> None:
        if (
            input.dim() == 0
            or input.dtype != self.dtype
            or input.device != self.device
            or not input.is_contiguous()
            or input.numel() <= 0
            or input.numel() > self.max_elems
        ):
            raise ValueError(
                "input must be a non-empty contiguous tensor with the workspace "
                f"dtype and at most {self.max_elems} elements on {self.device}"
            )

    def _validate_output(self, input: torch.Tensor, output: torch.Tensor) -> None:
        if (
            output.dtype != input.dtype
            or output.device != input.device
            or not output.is_contiguous()
            or output.numel() != input.numel() * self.world_size
        ):
            raise ValueError(
                "output must be contiguous, match input dtype/device, and contain "
                "world_size * input.numel() elements"
            )


def all_gather(
    input: torch.Tensor,
    workspace: SymmetricAllGatherWorkspace,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a symmetric-memory all-gather."""
    return workspace.all_gather(input, output)
