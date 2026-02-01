#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def _compute_mdd_abs(eq: pd.Series) -> float:
    eq = pd.to_numeric(eq, errors="coerce").dropna()
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return abs(float(dd.min()))

def _pick(df, names):
    for n in names:
        if n in df.columns: return n
    lc = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lc: return lc[n.lower()]
    return None

def main():
    root = Path("artifacts/reports")
    cfgp = Path("_kaggle_v71_wf_realistic.kaggle.yaml")
    cfg = yaml.safe_load(cfgp.read_text()) if cfgp.exists() else {}
    sim = cfg.get("sim", {}) if isinstance(cfg, dict) else {}
    fee_bps = float(sim.get("fee_bps", 0.0) or 0.0)
    slip_bps = float(sim.get("slippage_bps", 0.0) or 0.0)

    rows = []
    for fold_dir in sorted(root.glob("fold_*")):
        fold = fold_dir.name
        trades_path = fold_dir / "best_trades.csv"
        if not trades_path.exists():
            trades_path = fold_dir / "trades.csv"
        eq_path = fold_dir / "best_equity.csv"

        flags = []
        n_trades = 0
        net = 0.0
        pf = np.nan
        fee_sum = 0.0
        slip_sum = 0.0
        fee_est = 0.0
        slip_est = 0.0
        mdd = np.nan
        equity_final = np.nan

        if trades_path.exists():
            df = pd.read_csv(trades_path)
            n_trades = int(len(df))

            pnl_col = _pick(df, ["net_pnl","pnl","profit","net"])
            pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0) if pnl_col else pd.Series([0.0]*n_trades)
            net = float(pnl.sum())

            gp = float(pnl[pnl>0].sum())
            gl = float((-pnl[pnl<0]).sum())
            pf = float(gp/gl) if gl > 1e-12 else (np.inf if gp>0 else np.nan)

            fee_col = _pick(df, ["fee","fees","commission"])
            slip_col = _pick(df, ["slippage","slip","slippage_cost"])
            if fee_col: fee_sum = float(pd.to_numeric(df[fee_col], errors="coerce").fillna(0.0).sum())
            if slip_col: slip_sum = float(pd.to_numeric(df[slip_col], errors="coerce").fillna(0.0).sum())

            # estimación por notional si existen columnas
            qty_col = _pick(df, ["qty","size","contracts"])
            ep_col  = _pick(df, ["entry_price","px_entry"])
            xp_col  = _pick(df, ["exit_price","px_exit"])
            if qty_col and ep_col and xp_col:
                qty = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0).abs()
                ep  = pd.to_numeric(df[ep_col], errors="coerce").fillna(0.0).abs()
                xp  = pd.to_numeric(df[xp_col], errors="coerce").fillna(0.0).abs()
                notional_rt = qty*ep + qty*xp
                fee_est  = float((notional_rt * (fee_bps/1e4)).sum())
                slip_est = float((notional_rt * (slip_bps/1e4)).sum())

            if n_trades > 0 and slip_bps > 0 and slip_sum == 0.0:
                flags.append("SLIPPAGE_ZERO_WITH_TRADES")
            if n_trades > 0 and fee_bps > 0 and fee_sum == 0.0:
                flags.append("FEE_ZERO_WITH_TRADES")
        else:
            flags.append("NO_TRADES_FILE")

        if eq_path.exists():
            eq = pd.read_csv(eq_path)
            eq_col = _pick(eq, ["equity","balance","value"])
            if eq_col is None:
                numcols = [c for c in eq.columns if pd.api.types.is_numeric_dtype(eq[c])]
                eq_col = numcols[0] if numcols else None
            if eq_col:
                mdd = _compute_mdd_abs(eq[eq_col])
                equity_final = float(pd.to_numeric(eq[eq_col], errors="coerce").dropna().iloc[-1])
                if mdd > 1.0:
                    flags.append("MDD_GT_1")
        else:
            flags.append("NO_EQUITY_FILE")

        rows.append({
            "fold": fold,
            "n_trades": n_trades,
            "net_pnl": net,
            "pf": pf,
            "fee_sum": fee_sum,
            "slippage_sum": slip_sum,
            "fee_est_sum": fee_est,
            "slippage_est_sum": slip_est,
            "mdd": mdd,
            "equity_final": equity_final,
            "flags": "|".join(flags),
            "trades_file": str(trades_path) if trades_path.exists() else "",
            "equity_file": str(eq_path) if eq_path.exists() else "",
        })

    out = pd.DataFrame(rows)
    outp = root / "audit_post_wf_v2.csv"
    out.to_csv(outp, index=False)
    print("[OK] wrote", outp)
    print(out[["fold","n_trades","pf","net_pnl","fee_sum","slippage_sum","fee_est_sum","slippage_est_sum","mdd","equity_final","flags"]].to_string(index=False))

if __name__ == "__main__":
    main()
