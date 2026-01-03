import hashlib
import json
from pathlib import Path
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_hash_md5(cfg: dict) -> str:
    """
    Hash estable (ordenado) para detectar cambios de config.
    """
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(s).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
