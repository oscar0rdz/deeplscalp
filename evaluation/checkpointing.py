# DeepLScalp/evaluation/checkpointing.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import torch


def _atomic_json_write(path: str, obj: dict) -> None:
    """
    Escribe JSON de forma atómica: evita estados corruptos si se corta el proceso.
    """
    tmp = f"{path}.tmp_{int(time.time())}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def safe_json_load(path: str) -> dict | None:
    """
    Carga JSON; si está corrupto, lo renombra a .bad_TIMESTAMP y devuelve None.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        bad = f"{path}.bad_{int(time.time())}"
        try:
            os.replace(path, bad)
        except Exception:
            pass
        print(f"[WARN] state corrupto, se renombró a: {bad}. Se inicia sin resume. Error: {e}")
        return None


def save_checkpoint_pt(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    best_val: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        "extra": extra or {},
    }
    tmp = f"{ckpt_path}.tmp_{int(time.time())}"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)


def load_checkpoint_pt(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    device: str,
) -> dict | None:
    if not os.path.exists(ckpt_path):
        return None
    try:
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(payload["optimizer_state"])
            except Exception as e:
                print(f"[WARN] No se pudo cargar optimizer_state del ckpt; se reinicia optimizer. Motivo: {e}")
        if scheduler is not None and payload.get("scheduler_state") is not None and hasattr(scheduler, "load_state_dict"):
            try:
                scheduler.load_state_dict(payload["scheduler_state"])
            except Exception as e:
                print(f"[WARN] No se pudo cargar scheduler_state del ckpt; se reinicia scheduler. Motivo: {e}")
        return payload
    except Exception as e:
        bad = f"{ckpt_path}.bad_{int(time.time())}"
        try:
            os.replace(ckpt_path, bad)
        except Exception:
            pass
        print(f"[WARN] ckpt corrupto, se renombró a: {bad}. Se inicia sin resume. Error: {e}")
        return None


def write_state_json(
    state_path: str,
    fold_id: int,
    epoch: int,
    best_val: float,
    bad_epochs: int,
    ckpt_path: str,
) -> None:
    state = {
        "fold_id": int(fold_id),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "bad_epochs": int(bad_epochs),
        "ckpt_path": ckpt_path,
        "version": 2,
    }
    _atomic_json_write(state_path, state)
