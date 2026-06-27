import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _configure_source_jit_paths() -> None:
    from flashinfer.jit import env as jit_env

    if jit_env.FLASHINFER_CSRC_DIR.exists():
        return
    root = Path(__file__).resolve().parents[2]
    jit_env.FLASHINFER_CSRC_DIR = root / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = root / "include"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _expected(world_size: int, elems: int, offset: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.full(
                (elems,),
                rank + offset,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for rank in range(world_size)
        ]
    )


def _run_worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    workspace = None
    try:
        _configure_source_jit_paths()
        from flashinfer.comm import SymmetricAllGatherWorkspace
        from flashinfer.comm.mnnvl import TorchDistBackend

        backend = TorchDistBackend()
        elems = 4096
        workspace = SymmetricAllGatherWorkspace(
            max_elems=elems,
            world_size=world_size,
            rank=rank,
            comm_backend=backend,
            dtype=torch.bfloat16,
        )

        eager_input = torch.full(
            (elems,), rank + 1, dtype=torch.bfloat16, device="cuda"
        )
        eager_output = workspace.all_gather(eager_input)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager_output, _expected(world_size, elems, 1))

        static_input = torch.full(
            (elems,), rank + 4, dtype=torch.bfloat16, device="cuda"
        )
        static_output = torch.empty(
            world_size * elems, dtype=torch.bfloat16, device="cuda"
        )
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            workspace.all_gather(static_input, static_output)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_output, _expected(world_size, elems, 4))

        dist.barrier()
        workspace.prepare_checkpoint()
        dist.barrier()
        workspace.restore_after_checkpoint(TorchDistBackend())
        dist.barrier()

        static_input.fill_(rank + 8)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_output, _expected(world_size, elems, 8))
        assert workspace.status()[1:] == [0, 0, 0]
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="symmetric all-gather requires two CUDA devices",
)
def test_graph_replay_after_symmetric_memory_remap():
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer-tests")
    _configure_source_jit_paths()
    from flashinfer.comm.allgather import _get_module

    _get_module()
    mp.spawn(_run_worker, args=(2, _free_port()), nprocs=2, join=True)
