#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RECOVERY robusto de deeplscalp/modeling/train_v71.py

Objetivo:
- Si el archivo actual NO compila, recuperar automáticamente la versión más reciente
  del historial git que SÍ compile.
- Luego aplicar un parche mínimo para evitar KeyError en seq_len y otras claves.

Uso:
  python tools/recover_train_v71.py
"""

from __future__ import annotations
from pathlib import Path
import subprocess
import tempfile
import py_compile
import re
import sys


FILE = Path("deeplscalp/modeling/train_v71.py")


def sh(*args: str, check: bool = True) -> str:
    r = subprocess.run(args, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.stdout.strip()


def compiles_text(src_text: str) -> bool:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "train_v71.py"
        p.write_text(src_text, encoding="utf-8")
        try:
            py_compile.compile(str(p), doraise=True)
            return True
        except Exception:
            return False


def compiles_file(p: Path) -> bool:
    try:
        py_compile.compile(str(p), doraise=True)
        return True
    except Exception:
        return False


def ensure_history():
    # Si el repo es shallow, traer historial para poder buscar un commit “bueno”
    try:
        shallow = Path(".git/shallow")
        if shallow.exists():
            print("[recover] repo shallow → haciendo fetch --unshallow")
            subprocess.run(["git", "fetch", "--unshallow", "--tags"], check=False)
    except Exception:
        pass


def find_latest_good_commit() -> str | None:
    # commits que tocan el archivo
    try:
        out = sh("git", "rev-list", "--max-count=200", "HEAD", "--", str(FILE), check=True)
    except Exception:
        return None

    commits = [c for c in out.splitlines() if c.strip()]
    if not commits:
        return None

    # probar de más reciente a más viejo
    for sha in commits:
        try:
            src = sh("git", "show", f"{sha}:{FILE.as_posix()}", check=True)
        except Exception:
            continue
        if compiles_text(src):
            return sha
    return None


def restore_from_commit(sha: str):
    src = sh("git", "show", f"{sha}:{FILE.as_posix()}", check=True)
    FILE.write_text(src, encoding="utf-8")
    print(f"[recover] restaurado {FILE} desde commit {sha[:8]}")


def strip_bad_injections(text: str) -> str:
    # Elimina cualquier helper/marker previo y líneas de normalizador que suelen romper sintaxis
    # (Esto es un “último recurso” si no hay historial suficiente)
    # 1) Bloques marcados
    text = re.sub(r"(?s)^.*?#\s*---\s*V71_.*?---\s*end\s*V71_.*?---\s*\n", "", text, flags=re.MULTILINE)

    # 2) Líneas sueltas peligrosas
    text = re.sub(r"(?m)^\s*cfg\s*=\s*_v71_normalize_cfg\s*\(\s*cfg\s*\)\s*$\n?", "", text)
    text = re.sub(r"(?m)^\s*(tcfg|mcfg|fcfg)\s*=\s*cfg\[[^\]]+\]\s*$\n?", "", text)

    return text


def patch_safe_getters(text: str) -> str:
    """
    Parche mínimo: evita KeyError y NO mete normalizadores.
    """

    # predict_v71: seq_len seguro
    text = re.sub(
        r'''seq_len\s*=\s*int\(\s*cfg\["features"\]\["seq_len"\]\s*\)''',
        r'''seq_len = int((cfg.get("features") or {}).get("seq_len", 256))''',
        text
    )

    # train_model_v71: si hay accesos directos a cfg["train"] etc, hacerlos tolerantes
    # (sin reescribir toda la función; solo evita KeyError típicos)
    # Evitamos reemplazar asignaciones (LHS) usando negative lookahead
    # Patrón: cfg["algo"] NO seguido de =, +=, -=, etc.
    block_assign = r'''(?!\s*[\+\-\*\/%@&|^]?=)'''
    
    text = re.sub(
        r'''cfg\["features"\]''' + block_assign,
        r'''(cfg.get("features") or {})''',
        text
    )
    text = re.sub(
        r'''cfg\["train"\]''' + block_assign,
        r'''(cfg.get("train") or {})''',
        text
    )
    text = re.sub(
        r'''cfg\["model"\]''' + block_assign,
        r'''(cfg.get("model") or {})''',
        text
    )

    return text


def main() -> int:
    if not FILE.exists():
        print("[recover] no existe:", FILE)
        return 2

    ensure_history()

    # 1) Si ya compila, solo aplica parche mínimo de getters
    src0 = FILE.read_text(encoding="utf-8", errors="replace")
    if compiles_file(FILE):
        print("[recover] train_v71.py ya compila → aplicando parche mínimo (getters seguros)")
        src1 = patch_safe_getters(src0)
        FILE.write_text(src1, encoding="utf-8")
        py_compile.compile(str(FILE), doraise=True)
        print("[OK] compila ✅ y sin KeyError de seq_len")
        return 0

    # 2) Intentar restaurar desde el commit más reciente que compile
    sha = find_latest_good_commit()
    if sha:
        restore_from_commit(sha)
        src = FILE.read_text(encoding="utf-8", errors="replace")
        src = patch_safe_getters(src)
        FILE.write_text(src, encoding="utf-8")
        py_compile.compile(str(FILE), doraise=True)
        print("[OK] recovery+parche mínimo aplicado ✅")
        return 0

    # 3) Si no hay historial suficiente: limpiar inserciones y probar
    print("[recover] no encontré commit bueno en historial; intento limpieza de inserciones")
    cleaned = strip_bad_injections(src0)
    cleaned = patch_safe_getters(cleaned)
    FILE.write_text(cleaned, encoding="utf-8")

    py_compile.compile(str(FILE), doraise=True)
    print("[OK] limpieza+parche mínimo aplicado ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
