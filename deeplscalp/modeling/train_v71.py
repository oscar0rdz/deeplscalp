from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from training.itransformer_v71 import ITransformerV71, ITransV71Config, quantile_loss


@dataclass
class StandardScaler:
    mean_: np.ndarray
    std_: np.ndarray

    def transform2d(self, x2d: np.ndarray) -> np.ndarray:
        return (x2d - self.mean_) / (self.std_ + 1e-12)

    def fit_from2d(self, x2d: np.ndarray) -> "StandardScaler":
        self.mean_ = x2d.mean(axis=0)
        self.std_ = x2d.std(axis=0)
        self.std_ = np.maximum(self.std_, 1e-6)
        return self


def _make_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    X = train_df[feature_cols].astype("float32").values
    scaler = StandardScaler(
        mean_=np.zeros(X.shape[1], dtype=np.float32),
        std_=np.ones(X.shape[1], dtype=np.float32),
    )
    return scaler.fit_from2d(X)


class SeqDatasetV71(Dataset):
    """
    Dataset secuencial sin construir tensores gigantes:
      - almacena X_scaled 2D (N,F)
      - extrae ventanas (T,F) por slicing contiguo
    """
    def __init__(self, df: pd.DataFrame, X_scaled_2d: np.ndarray, feature_cols: list[str], seq_len: int):
        self.df = df
        self.X = X_scaled_2d.astype("float32", copy=False)
        self.seq_len = int(seq_len)

        # labels
        self.y_side = df["y_side"].astype("int64").values
        self.yc_long = df["yc_long"].astype("int64").values
        self.yc_short = df["yc_short"].astype("int64").values
        self.yL_reg = df["yL_reg"].astype("float32").values
        self.yS_reg = df["yS_reg"].astype("float32").values
        self.y_regime = df["y_regime"].astype("int64").values
        self.y_event = df["y_event"].astype("int64").values

        self.sample_weight = (
            df["sample_weight"].astype("float32").values
            if "sample_weight" in df.columns else np.ones(len(df), dtype="float32")
        )
        self.is_event = (
            df["is_event"].astype("float32").values
            if "is_event" in df.columns else np.zeros(len(df), dtype="float32")
        )

        # índices válidos para seq
        idx = np.arange(len(df), dtype=np.int64)
        self.idx = idx[idx >= (self.seq_len - 1)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int):
        t = int(self.idx[i])
        x_seq = self.X[t - self.seq_len + 1: t + 1]  # (T,F) view contigua

        return (
            torch.from_numpy(x_seq),
            torch.tensor(self.y_side[t], dtype=torch.long),
            torch.tensor(self.yc_long[t], dtype=torch.long),
            torch.tensor(self.yc_short[t], dtype=torch.long),
            torch.tensor(self.yL_reg[t], dtype=torch.float32),
            torch.tensor(self.yS_reg[t], dtype=torch.float32),
            torch.tensor(self.y_regime[t], dtype=torch.long),
            torch.tensor(self.y_event[t], dtype=torch.long),
            torch.tensor(self.sample_weight[t], dtype=torch.float32),
            torch.tensor(self.is_event[t], dtype=torch.float32),
        )


class InferenceSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, X_scaled_2d: np.ndarray, seq_len: int):
        self.df = df
        self.X = X_scaled_2d.astype("float32", copy=False)
        self.seq_len = int(seq_len)

        idx = np.arange(len(df), dtype=np.int64)
        self.idx = idx[idx >= (self.seq_len - 1)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int):
        t = int(self.idx[i])
        x_seq = self.X[t - self.seq_len + 1: t + 1]
        return torch.from_numpy(x_seq), t


def train_model_v71(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], cfg: dict,
                    device: str, fold_id: int, out_dir: Path):

    os.makedirs(out_dir, exist_ok=True)

    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32

    cpu_threads = int(cfg.get("cpu_threads", max(2, (os.cpu_count() or 6) // 2)))
    torch.set_num_threads(cpu_threads)
    # interop threads puede ayudar en CPU
    try:
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
    except Exception:
        pass

    seq_len = int(cfg["features"]["seq_len"])
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    quantiles = [float(q) for q in mcfg["quantiles"]]

    # drop nan
    train_df = train_df.dropna()
    val_df = val_df.dropna()

    # scaler + pre-escalado 2D (clave para velocidad)
    scaler = _make_scaler(train_df, feature_cols)

    Xtr = train_df[feature_cols].astype("float32").values
    Xva = val_df[feature_cols].astype("float32").values
    Xtr_s = scaler.transform2d(Xtr)
    Xva_s = scaler.transform2d(Xva)

    ds_train = SeqDatasetV71(train_df, Xtr_s, feature_cols, seq_len=seq_len)
    ds_val = SeqDatasetV71(val_df, Xva_s, feature_cols, seq_len=seq_len)

    # GPU-optimized DataLoader
    workers = int(tcfg.get("num_workers", cfg.get("dataloader_workers", 0)))
    if workers < 0:
        workers = 0

    pin_memory = bool(tcfg.get("pin_memory", device.startswith("cuda")))
    persistent_workers = bool(tcfg.get("persistent_workers", workers > 0))
    prefetch_factor = int(tcfg.get("prefetch_factor", 2 if workers > 0 else None))

    dl_kwargs = {
        "batch_size": int(tcfg["batch_size"]),
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor and workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    dl_train = DataLoader(
        ds_train,
        shuffle=True,
        drop_last=True,
        **dl_kwargs
    )
    dl_val = DataLoader(
        ds_val,
        shuffle=False,
        drop_last=False,
        **dl_kwargs
    )

    model_cfg = ITransV71Config(
        seq_len=seq_len,
        n_features=len(feature_cols),
        d_model=int(mcfg.get("d_model", 128)),
        nhead=int(mcfg.get("nhead", 4)),
        num_layers=int(mcfg.get("num_layers", 2)),
        dropout=float(mcfg.get("dropout", 0.10)),
        quantiles=tuple(quantiles),
        norm_first=bool(mcfg.get("norm_first", False)),
    )
    model = ITransformerV71(model_cfg).to(device)

    # class weights
    side_w = torch.tensor(tcfg.get("side_class_weights", [1.0, 2.0, 2.0]), dtype=torch.float32, device=device)
    hit_w = torch.tensor(tcfg.get("hit_class_weights", [2.0, 1.0, 2.0]), dtype=torch.float32, device=device)
    reg_w = torch.tensor(tcfg.get("regime_class_weights", [1.2, 1.0, 1.0, 1.5]), dtype=torch.float32, device=device)
    evt_w = torch.tensor(tcfg.get("event_class_weights", [1.0, 1.3, 1.3, 1.8]), dtype=torch.float32, device=device)

    ce_side = torch.nn.CrossEntropyLoss(weight=side_w, reduction="none")
    ce_hit = torch.nn.CrossEntropyLoss(weight=hit_w, reduction="none")
    ce_reg = torch.nn.CrossEntropyLoss(weight=reg_w, reduction="none")
    ce_evt = torch.nn.CrossEntropyLoss(weight=evt_w, reduction="none")

    reg_alpha = float(tcfg.get("reg_alpha", 1.0))
    cls_alpha_side = float(tcfg.get("cls_alpha_side", 1.0))
    cls_alpha_hit = float(tcfg.get("cls_alpha_hit", 0.8))
    cls_alpha_regime = float(tcfg.get("cls_alpha_regime", 0.5))
    cls_alpha_event = float(tcfg.get("cls_alpha_event", 0.5))

    event_weight = float(tcfg.get("event_weight", 1.5))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("wd", 1e-4)),
    )

    log_every = int(tcfg.get("log_every_steps", 200))
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    # AMP setup
    amp_enabled = bool(tcfg.get("amp", False))
    scaler = GradScaler(enabled=amp_enabled)

    def run_epoch(dl, train: bool, ep: int):
        model.train(train)
        losses = []
        t0 = time.time()
        seen = 0

        for step, batch in enumerate(dl, start=1):
            X, y_side, ycL, ycS, yL, yS, y_reg, y_evt, sw, is_evt = batch

            X = X.to(device)
            y_side = y_side.to(device)
            ycL = ycL.to(device)
            ycS = ycS.to(device)
            yL = yL.to(device)
            yS = yS.to(device)
            y_reg = y_reg.to(device)
            y_evt = y_evt.to(device)
            sw = sw.to(device)
            is_evt = is_evt.to(device)

            if train:
                opt.zero_grad(set_to_none=True)

            # AMP forward pass
            with autocast(enabled=amp_enabled):
                side_logits, hitL_logits, hitS_logits, qL, qS, reg_logits, evt_logits, emb = model(X)

                w = sw * (1.0 + event_weight * is_evt)

                lqL = quantile_loss(qL, yL, quantiles, reduction="none")
                lqS = quantile_loss(qS, yS, quantiles, reduction="none")
                l_reg = (0.5 * (lqL + lqS) * w).mean()

                l_side = (ce_side(side_logits, y_side) * w).mean()
                l_hit = (0.5 * (ce_hit(hitL_logits, ycL) + ce_hit(hitS_logits, ycS)) * w).mean()
                l_regime = (ce_reg(reg_logits, y_reg) * w).mean()
                l_event = (ce_evt(evt_logits, y_evt) * w).mean()

                loss = reg_alpha * l_reg + cls_alpha_side * l_side + cls_alpha_hit * l_hit + cls_alpha_regime * l_regime + cls_alpha_event * l_event

            if train:
                # AMP backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()

            losses.append(float(loss.detach().cpu().item()))
            seen += int(X.shape[0])

            if log_every > 0 and (step % log_every == 0):
                dt = max(1e-6, time.time() - t0)
                rate = seen / dt
                avg = float(np.mean(losses[-log_every:]))
                mode = "train" if train else "val"
                print(f"[v71] ep={ep} {mode} step={step}/{len(dl)} avg_loss={avg:.6f} samp_per_s={rate:.1f}")

        return float(np.mean(losses)) if losses else 0.0

    epochs = int(tcfg["epochs"])
    patience = int(tcfg.get("early_stop_patience", 4))
    best_val = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        tr = run_epoch(dl_train, True, ep)

        with torch.inference_mode():
            va = run_epoch(dl_val, False, ep)

        print(f"[v71] ep={ep} loss_train={tr:.6f} loss_val={va:.6f}")

        if va < best_val - 1e-6:
            best_val = va
            bad = 0
            ckpt = out_dir / f"fold_{fold_id}_best.pt"
            torch.save({"model": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_std": scaler.std_}, ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"[v71] early stop at ep={ep}")
                break

    ckpt = out_dir / f"fold_{fold_id}_best.pt"
    obj = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(obj["model"])
    scaler.mean_ = obj["scaler_mean"]
    scaler.std_ = obj["scaler_std"]
    model.to(device).eval()

    return model, scaler


def predict_v71(model, scaler, df: pd.DataFrame, feature_cols: list[str], cfg: dict, device: str) -> pd.DataFrame:
    model.eval()
    seq_len = int(cfg["features"]["seq_len"])
    quantiles = [float(q) for q in cfg["model"]["quantiles"]]

    df = df.dropna()
    X = df[feature_cols].astype("float32").values
    Xs = scaler.transform2d(X)

    ds = InferenceSeqDataset(df, Xs, seq_len=seq_len)
    workers = int(cfg.get("dataloader_workers", 0))
    if workers < 0:
        workers = 0

    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        persistent_workers=(workers > 0),
    )

    side_all, hitL_all, hitS_all, reg_all, evt_all = [], [], [], [], []
    qL_all, qS_all, t_all = [], [], []

    with torch.inference_mode():
        for Xb, ts in dl:
            Xb = Xb.to(device)
            side_logits, hitL_logits, hitS_logits, qL, qS, reg_logits, evt_logits, emb = model(Xb)
            side_all.append(torch.softmax(side_logits, dim=1).cpu().numpy())
            hitL_all.append(torch.softmax(hitL_logits, dim=1).cpu().numpy())
            hitS_all.append(torch.softmax(hitS_logits, dim=1).cpu().numpy())
            reg_all.append(torch.softmax(reg_logits, dim=1).cpu().numpy())
            evt_all.append(torch.softmax(evt_logits, dim=1).cpu().numpy())
            qL_all.append(qL.cpu().numpy())
            qS_all.append(qS.cpu().numpy())
            t_all.append(ts.numpy())

    side_all = np.concatenate(side_all)
    hitL_all = np.concatenate(hitL_all)
    hitS_all = np.concatenate(hitS_all)
    reg_all = np.concatenate(reg_all)
    evt_all = np.concatenate(evt_all)
    qL_all = np.concatenate(qL_all)
    qS_all = np.concatenate(qS_all)
    t_all = np.concatenate(t_all)

    idx = df.index[t_all]
    out = pd.DataFrame(index=idx)

    out["p_flat"] = side_all[:, 0].astype("float32")
    out["p_long"] = side_all[:, 1].astype("float32")
    out["p_short"] = side_all[:, 2].astype("float32")

    out["pL_sl"] = hitL_all[:, 0].astype("float32")
    out["pL_time"] = hitL_all[:, 1].astype("float32")
    out["pL_tp"] = hitL_all[:, 2].astype("float32")

    out["pS_sl"] = hitS_all[:, 0].astype("float32")
    out["pS_time"] = hitS_all[:, 1].astype("float32")
    out["pS_tp"] = hitS_all[:, 2].astype("float32")

    out["p_reg_range"] = reg_all[:, 0].astype("float32")
    out["p_reg_up"] = reg_all[:, 1].astype("float32")
    out["p_reg_dn"] = reg_all[:, 2].astype("float32")
    out["p_reg_spike"] = reg_all[:, 3].astype("float32")

    out["p_evt_none"] = evt_all[:, 0].astype("float32")
    out["p_evt_breakout"] = evt_all[:, 1].astype("float32")
    out["p_evt_rebound"] = evt_all[:, 2].astype("float32")
    out["p_evt_spike"] = evt_all[:, 3].astype("float32")

    for j, q in enumerate(quantiles):
        out[f"qL{int(q*100):02d}_reg"] = qL_all[:, j].astype("float32")
        out[f"qS{int(q*100):02d}_reg"] = qS_all[:, j].astype("float32")

    # desescalado a gross (si existe vol_scale)
    if bool(cfg["labels"].get("vol_scaled", True)) and "vol_scale" in df.columns:
        vs = df.loc[idx, "vol_scale"].astype("float32")
        out["vol_scale"] = vs.values
        for qq in [10, 50, 90]:
            out[f"qL{qq:02d}_gross"] = (out[f"qL{qq:02d}_reg"] * out["vol_scale"]).astype("float32")
            out[f"qS{qq:02d}_gross"] = (out[f"qS{qq:02d}_reg"] * out["vol_scale"]).astype("float32")
        out["iqrL_gross"] = (out["qL90_gross"] - out["qL10_gross"]).abs().astype("float32")
        out["iqrS_gross"] = (out["qS90_gross"] - out["qS10_gross"]).abs().astype("float32")
    else:
        for qq in [10, 50, 90]:
            out[f"qL{qq:02d}_gross"] = out[f"qL{qq:02d}_reg"].astype("float32")
            out[f"qS{qq:02d}_gross"] = out[f"qS{qq:02d}_reg"].astype("float32")
        out["iqrL_gross"] = (out["qL90_gross"] - out["qL10_gross"]).abs().astype("float32")
        out["iqrS_gross"] = (out["qS90_gross"] - out["qS10_gross"]).abs().astype("float32")

    # precios y ATR para sim
    out["atr"] = df.loc[idx, "atr"].astype("float32").values if "atr" in df.columns else 0.0
    out["open"] = df.loc[idx, "open"].astype("float32").values
    out["open_next"] = df["open"].shift(-1).loc[idx].astype("float32").values
    out["close"] = df.loc[idx, "close"].astype("float32").values
    out["high"] = df.loc[idx, "high"].astype("float32").values
    out["low"] = df.loc[idx, "low"].astype("float32").values

    # set datetime index
    out.index = pd.to_datetime(df.loc[idx, "ds"], utc=True)
    return out.sort_index()
