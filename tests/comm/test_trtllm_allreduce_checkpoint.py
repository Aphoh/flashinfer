import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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
        import flashinfer.comm as comm
        from flashinfer.comm.mnnvl import TorchDistBackend

        backend = TorchDistBackend()
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=1,
            hidden_dim=4096,
            dtype=torch.bfloat16,
            comm_backend=backend,
            checkpointable=True,
        )
        input_ = torch.full(
            (1, 4096), rank + 1, dtype=torch.bfloat16, device="cuda"
        )
        output = torch.empty_like(input_)

        def all_reduce() -> None:
            comm.allreduce_fusion(
                input=input_,
                workspace=workspace,
                output=output,
                pattern=comm.AllReduceFusionPattern.kAllReduce,
                use_oneshot=True,
            )

        all_reduce()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 3))

        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            all_reduce()

        dist.barrier()
        workspace.prepare_checkpoint()
        dist.barrier()
        workspace.restore_after_checkpoint(TorchDistBackend())
        dist.barrier()

        input_.fill_(rank + 2)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 5))
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="checkpointable TRT-LLM all-reduce requires two CUDA devices",
)
def test_graph_replay_after_symmetric_memory_remap():
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer-tests")
    mp.spawn(_run_worker, args=(2, _free_port()), nprocs=2, join=True)
