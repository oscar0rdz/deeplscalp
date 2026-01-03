import os
import torch


def set_torch_threads(n: int) -> None:
    try:
        torch.set_num_threads(max(1, int(n)))
        print(f"[cpu] torch threads={torch.get_num_threads()}")
    except Exception:
        pass


def pick_device(cfg_device: str) -> str:
    if cfg_device != "auto":
        return cfg_device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"

    return "cpu"
