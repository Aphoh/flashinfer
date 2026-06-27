from types import SimpleNamespace

import pytest

import flashinfer.comm.allreduce as allreduce
from flashinfer.comm.checkpoint import (
    SymmetricMemoryCheckpoint,
    SymmetricMemoryState,
)


class FakeDriver:
    class CUresult:
        CUDA_SUCCESS = 0

    def __init__(self):
        self.calls = []
        self.fail_once = set()

    def _call(self, name, *args):
        self.calls.append((name, args))
        if name in self.fail_once:
            self.fail_once.remove(name)
            return (1,)
        return (0,)

    def cuMemUnmap(self, *args):
        return self._call("cuMemUnmap", *args)

    def cuMemRelease(self, *args):
        return self._call("cuMemRelease", *args)

    def cuMemMap(self, *args):
        return self._call("cuMemMap", *args)

    def cuMemSetAccess(self, *args):
        return self._call("cuMemSetAccess", *args)

    def cuMemAddressFree(self, *args):
        return self._call("cuMemAddressFree", *args)

    def cuMemFree(self, *args):
        return self._call("cuMemFree", *args)

    def cuMemsetD8(self, *args):
        return self._call("cuMemsetD8", *args)


class FakeExchanger:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeBackend:
    def __init__(self):
        self.barriers = 0

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 2

    def barrier(self):
        self.barriers += 1


def make_memory(*, handles=(11, 12), ptrs=(100, 200)):
    return SimpleNamespace(
        buf_size=128,
        group_size=2,
        group_rank=0,
        device_idx=0,
        allocation_size=64,
        total_uc_size=128,
        uc_base_ptr=ptrs[0],
        uc_ptrs=list(ptrs),
        uc_handles=list(handles),
        uc_ptrs_dev=500,
        mc_handle=0,
        mc_ptr=0,
        comm_backend=object(),
        _exchanger=FakeExchanger(),
        _get_mem_access_desc=lambda: object(),
    )


def test_detach_can_resume_after_driver_failure():
    driver = FakeDriver()
    driver.fail_once.add("cuMemRelease")
    memory = make_memory()
    checkpoint = SymmetricMemoryCheckpoint(memory, _driver_factory=lambda: driver)

    with pytest.raises(RuntimeError):
        checkpoint.detach()

    assert checkpoint.state is SymmetricMemoryState.DETACHING
    checkpoint.detach()
    assert checkpoint.state is SymmetricMemoryState.DETACHED
    assert memory.uc_handles == [0, 0]
    unmapped = [args[0] for name, args in driver.calls if name == "cuMemUnmap"]
    assert unmapped.count(100) == 1


def test_failed_restore_rolls_back_and_can_retry():
    driver = FakeDriver()
    memory = make_memory()
    fresh_memories = []

    def memory_factory(**_kwargs):
        fresh = make_memory(handles=(21, 22), ptrs=(300, 400))
        fresh_memories.append(fresh)
        return fresh

    checkpoint = SymmetricMemoryCheckpoint(
        memory,
        _driver_factory=lambda: driver,
        _memory_factory=memory_factory,
    )
    checkpoint.detach()
    driver.fail_once.add("cuMemMap")

    with pytest.raises(RuntimeError):
        checkpoint.restore(FakeBackend())

    assert checkpoint.state is SymmetricMemoryState.DETACHED
    assert fresh_memories[0].uc_handles == []

    replacement_backend = FakeBackend()
    checkpoint.restore(replacement_backend)
    assert checkpoint.state is SymmetricMemoryState.ATTACHED
    assert memory.uc_handles == [21, 22]
    assert memory.comm_backend is replacement_backend


def test_restore_rejects_changed_group_shape():
    memory = make_memory()
    checkpoint = SymmetricMemoryCheckpoint(memory, _driver_factory=FakeDriver)
    checkpoint.detach()
    backend = FakeBackend()
    backend.Get_size = lambda: 4

    with pytest.raises(ValueError, match="size changed"):
        checkpoint.restore(backend)


def test_allreduce_workspace_reset_failure_can_retry(monkeypatch):
    class FakeCheckpoint:
        attached = True

        def __init__(self):
            self.restore_calls = 0

        def detach(self):
            self.attached = False

        def restore(self, _backend):
            self.restore_calls += 1
            self.attached = True

    workspace = object.__new__(allreduce.TRTLLMAllReduceFusionWorkspace)
    workspace._destroyed = False
    workspace.ipc_handles = [[100], [200], [300]]
    workspace.workspace_tensor = object()
    workspace.mem_handles = []
    workspace.metadata = {}
    checkpoint = FakeCheckpoint()
    workspace._memory_checkpoints = [checkpoint]
    workspace._checkpoint_workspace_ptrs = None
    monkeypatch.setattr(allreduce.torch.cuda, "synchronize", lambda: None)

    workspace.prepare_checkpoint()
    assert workspace.checkpoint_detached

    reset_calls = 0

    def reset(*_args):
        nonlocal reset_calls
        reset_calls += 1
        if reset_calls == 1:
            raise RuntimeError("injected reset failure")

    monkeypatch.setattr(
        allreduce,
        "trtllm_reset_ipc_workspace_for_all_reduce_fusion",
        reset,
    )
    backend = FakeBackend()
    with pytest.raises(RuntimeError, match="injected reset failure"):
        workspace.restore_after_checkpoint(backend)

    workspace.restore_after_checkpoint(backend)
    assert checkpoint.restore_calls == 2
    assert reset_calls == 2
    assert backend.barriers == 1
    assert workspace._checkpoint_workspace_ptrs is None
