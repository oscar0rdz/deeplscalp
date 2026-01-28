import os
import torch


def set_torch_threads(n: int) -> None:
    try:
        torch.set_num_threads(max(1, int(n)))
        print(f"[cpu] torch threads={torch.get_num_threads()}")
    except Exception:
        pass


def pick_device(cfg_device: str = "auto") -> torch.device:
    """
    Selecciona el mejor dispositivo disponible (CUDA > MPS > CPU).
    """
    if cfg_device != "auto" and cfg_device is not None:
        return torch.device(cfg_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS fallback ayuda con operaciones no soportadas
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")

    return torch.device("cpu")
