#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hotfix idempotente para deeplscalp/modeling/train_v71.py

Arregla:
- SyntaxError introducido por inserciones previas de cfg = _v71_normalize_cfg(cfg)
- Asegura defaults mínimos en cfg (features.seq_len, model.quantiles, train.*)
- Inserta llamadas a normalizador al inicio de train_model_v71() y predict_v71()
- Verifica compilación con py_compile

Uso:
  python tools/patch_train_v71_hotfix.py
"""

from __future__ import annotations
from pathlib import Path
import re
import py_compile
import sys


MARK = "V71_CFG_NORMALIZER_HOTFIX_V3"


def _find_def_block(lines: list[str], def_name: str) -> tuple[int, int] | None:
    """
    Devuelve (i_def, i_body_start) donde i_body_start es la primera línea del cuerpo
    (después de def y líneas vacías). No parsea AST (funciona aunque el archivo esté roto).
    Maneja definiciones multilínea buscando el '):' final.
    """
    pat = re.compile(rf"^\s*def\s+{re.escape(def_name)}\s*\(")
    i_def = None
    for i, ln in enumerate(lines):
        if pat.match(ln):
            i_def = i
            break
    if i_def is None:
        return None

    # Buscar el final de la firma (línea que termina en :)
    # Esto es heurístico pero más robusto que asumir una sola línea
    j = i_def
    citation_open = False # ' or "
    parenthesis_depth = 0
    
    # Simple scan forward for the ending colon of the def
    while j < len(lines):
        ln = lines[j].strip()
        # Si termina en : y los paréntesis están balanceados (simple heuristic)
        # Ojo: esto puede fallar con comentarios o strings raros, pero es un hotfix.
        # Mejor heurística: buscar '):' al final, ignorando trailing comments/whitespaces
        if ln.endswith(":") and "):" in "".join(lines[i_def : j+1]).replace("\n", ""):
             # Check basic match
             pass
        
        if ln.rstrip().endswith(":"):
            # Asumimos que es el final del def
            break
        j += 1
    
    if j >= len(lines):
        # Fallback si no encontramos ':', asumimos línea siguiente a i_def (comportamiento original pero malo)
        # pero mejor retornar i_def + 1 y que sea lo que desa
        j = i_def

    body_start = j + 1
    while body_start < len(lines) and lines[body_start].strip() == "":
        body_start += 1
        
    return i_def, body_start


def _skip_docstring(lines: list[str], i: int) -> int:
    """
    Si en lines[i] inicia docstring triple-comillas, salta hasta cierre y retorna el índice posterior.
    """
    if i >= len(lines):
        return i
    s = lines[i].lstrip()
    if not (s.startswith('"""') or s.startswith("'''")):
        return i

    q = '"""' if s.startswith('"""') else "'''"

    # docstring de una línea
    if s.count(q) >= 2 and s.rstrip().endswith(q):
        return i + 1

    # docstring multilínea
    j = i + 1
    while j < len(lines):
        if q in lines[j]:
            return j + 1
        j += 1
    return i + 1  # fallback


def _insert_after(lines: list[str], idx: int, new_lines: list[str]) -> list[str]:
    return lines[:idx] + new_lines + lines[idx:]


def _ensure_module_helper(lines: list[str]) -> list[str]:
    txt = "".join(lines)
    if MARK in txt:
        return lines

    # insertar helper antes de la primera def train_model_v71(
    anchor = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*def\s+train_model_v71\s*\(", ln):
            anchor = i
            break
    if anchor is None:
        raise SystemExit("No encontré def train_model_v71(...) en train_v71.py")

    helper = [
        f"# --- {MARK} ---\n",
        "def _v71_normalize_cfg(cfg: dict | None) -> dict:\n",
        "    \"\"\"Normaliza cfg para evitar KeyError y defaults inconsistentes.\"\"\"\n",
        "    if cfg is None:\n",
        "        cfg = {}\n",
        "\n",
        "    f = cfg.get('features') or {}\n",
        "    if not isinstance(f, dict):\n",
        "        raise TypeError(f\"cfg['features'] debe ser dict, no {type(f).__name__}\")\n",
        "    f.setdefault('seq_len', 256)\n",
        "    cfg['features'] = f\n",
        "\n",
        "    m = cfg.get('model') or {}\n",
        "    if not isinstance(m, dict):\n",
        "        raise TypeError(f\"cfg['model'] debe ser dict, no {type(m).__name__}\")\n",
        "    m.setdefault('quantiles', [0.1, 0.5, 0.9])\n",
        "    cfg['model'] = m\n",
        "\n",
        "    t = cfg.get('train') or {}\n",
        "    if not isinstance(t, dict):\n",
        "        raise TypeError(f\"cfg['train'] debe ser dict, no {type(t).__name__}\")\n",
        "    t.setdefault('batch_size', 256)\n",
        "    t.setdefault('epochs', 3)\n",
        "    t.setdefault('lr', 3e-4)\n",
        "    t.setdefault('weight_decay', 0.0)\n",
        "    t.setdefault('grad_clip', 1.0)\n",
        "    t.setdefault('workers', 2)\n",
        "    t.setdefault('prefetch_factor', 2)\n",
        "    t.setdefault('seed', 7)\n",
        "    # compat:\n",
        "    t.setdefault('num_workers', t.get('workers', 2))\n",
        "    cfg['train'] = t\n",
        "\n",
        "    return cfg\n",
        f"# --- end {MARK} ---\n\n",
    ]
    return _insert_after(lines, anchor, helper)


def _remove_stray_normalize_assignments(lines: list[str]) -> list[str]:
    """
    Elimina TODAS las líneas sueltas tipo:
      cfg = _v71_normalize_cfg(cfg)
    porque si una quedó insertada dentro de paréntesis/llamadas, rompe sintaxis.
    Luego reinsertamos en el lugar correcto (inicio de funciones).
    """
    out = []
    pat = re.compile(r"^\s*cfg\s*=\s*_v71_normalize_cfg\s*\(\s*cfg\s*\)\s*$")
    removed = 0
    for ln in lines:
        if pat.match(ln.rstrip("\n")):
            removed += 1
            continue
        out.append(ln)
    if removed:
        print(f"[hotfix] removidas {removed} líneas sueltas 'cfg = _v71_normalize_cfg(cfg)'")
    return out


def _inject_at_top_of_def(lines: list[str], def_name: str, payload: list[str]) -> list[str]:
    blk = _find_def_block(lines, def_name)
    if blk is None:
        raise SystemExit(f"No encontré def {def_name}(...) en train_v71.py")
    i_def, i_body = blk

    # detectar indent del cuerpo
    if i_body >= len(lines):
        return lines
    indent = re.match(r"^\s*", lines[i_body]).group(0)

    # saltar docstring si existe
    i_ins = _skip_docstring(lines, i_body)

    # ventana para evitar duplicado
    window = "".join(lines[i_def : min(len(lines), i_def + 60)])
    if "_v71_normalize_cfg(cfg)" in window:
        # ya está insertado correctamente en algún punto cercano
        return lines

    inject = [indent + x + "\n" for x in payload]
    return _insert_after(lines, i_ins, inject)


def main() -> int:
    p = Path("deeplscalp/modeling/train_v71.py")
    if not p.exists():
        print("No existe:", p)
        return 2

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines(True)

    # 1) remover inserciones peligrosas
    lines = _remove_stray_normalize_assignments(lines)

    # 2) asegurar helper a nivel módulo
    lines = _ensure_module_helper(lines)

    # 3) inyectar normalización al inicio de funciones
    lines = _inject_at_top_of_def(
        lines,
        "train_model_v71",
        [
            "cfg = _v71_normalize_cfg(cfg)",
            "tcfg = cfg['train']",
            "mcfg = cfg['model']",
            "fcfg = cfg['features']",
        ],
    )
    lines = _inject_at_top_of_def(
        lines,
        "predict_v71",
        [
            "cfg = _v71_normalize_cfg(cfg)",
        ],
    )

    p.write_text("".join(lines), encoding="utf-8")

    # 4) compilar para asegurar que ya no hay SyntaxError
    py_compile.compile(str(p), doraise=True)
    print("[OK] train_v71.py hotfix aplicado y compila ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
