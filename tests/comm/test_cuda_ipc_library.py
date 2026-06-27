import io
from types import SimpleNamespace

import flashinfer.comm.cuda_ipc as cuda_ipc


def test_find_loaded_library_skips_stub_and_missing_symbols(monkeypatch):
    maps = io.StringIO(
        "0-1 r-xp 0 00:00 0 /opt/tilelang/lib/libcudart_stub.so\n"
        "1-2 r-xp 0 00:00 0 /tmp/libcudart_incomplete.so.13\n"
        "2-3 r-xp 0 00:00 0 /usr/local/cuda/lib64/libcudart.so.13\n"
    )
    libraries = {
        "/tmp/libcudart_incomplete.so.13": SimpleNamespace(cudaMalloc=object()),
        "/usr/local/cuda/lib64/libcudart.so.13": SimpleNamespace(
            cudaMalloc=object(),
            cudaDeviceReset=object(),
        ),
    }
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: maps)
    monkeypatch.setattr(cuda_ipc.ctypes, "CDLL", libraries.__getitem__)

    assert (
        cuda_ipc.find_loaded_library("libcudart", ["cudaMalloc", "cudaDeviceReset"])
        == "/usr/local/cuda/lib64/libcudart.so.13"
    )
