# deeplscalp/modeling/train.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from training.itransformer_v7 import ITransformerV7


# -------------------------
# Utils
# -------------------------
@dataclass
class StandardScaler:
    mean_: np.ndarray
    std_: np.ndarray

    def fit_from(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / (self.std_ + 1e-12)


def _make_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    X = train_df[feature_cols].astype("float32").values
    scaler = StandardScaler(
        mean_=np.zeros(X.shape[1], dtype=np.float32),
        std_=np.ones(X.shape[1], dtype=np.float32),
    )
    return scaler.fit_from(X)


def quantile_loss(q_pred: torch.Tensor, y: torch.Tensor, quantiles: list[float], reduction: str = "mean"):
    """
    q_pred: [B,Q], y: [B]
    """
    qs = torch.tensor(quantiles, device=q_pred.device, dtype=q_pred.dtype).view(1, -1)
    y = y.view(-1, 1)
    e = y - q_pred
    loss = torch.maximum(qs * e, (qs - 1.0) * e)  # pinball
    loss = loss.mean(dim=1)  # (B,)
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'mean' or 'none'")


# -------------------------
# Datasets
# -------------------------
class SeqDatasetV7(Dataset):
    """
    Espera columnas:
      y_side (0/1/2)
      yc_long (0/1/2)
      yc_short (0/1/2)
      yL_reg (float)
      yS_reg (float)
      sample_weight (float)
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], seq_len: int):
        self.df = df
        self.feature_cols = feature_cols
        self.seq_len = int(seq_len)

        self.x = df[feature_cols].astype("float32").values

        self.y_side = df["y_side"].astype("int64").values
        self.yc_long = df["yc_long"].astype("int64").values
        self.yc_short = df["yc_short"].astype("int64").values
        self.yL_reg = df["yL_reg"].astype("float32").values
        self.yS_reg = df["yS_reg"].astype("float32").values

        if "sample_weight" in df.columns:
            self.sample_weight = df["sample_weight"].astype("float32").values
        else:
            self.sample_weight = np.ones(len(df), dtype="float32")

        self.idx = np.arange(len(df))
        self.idx = self.idx[self.idx >= (self.seq_len - 1)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = int(self.idx[i])
        x_seq = self.x[t - self.seq_len + 1: t + 1]
        return (
            torch.from_numpy(x_seq),
            torch.tensor(self.y_side[t], dtype=torch.long),
            torch.tensor(self.yc_long[t], dtype=torch.long),
            torch.tensor(self.yc_short[t], dtype=torch.long),
            torch.tensor(self.yL_reg[t], dtype=torch.float32),
            torch.tensor(self.yS_reg[t], dtype=torch.float32),
            torch.tensor(self.sample_weight[t], dtype=torch.float32),
        )


class InferenceSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], seq_len: int):
        self.df = df
        self.feature_cols = feature_cols
        self.seq_len = int(seq_len)
        self.x = df[feature_cols].astype("float32").values
        self.idx = np.arange(len(df))
        self.idx = self.idx[self.idx >= (self.seq_len - 1)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = int(self.idx[i])
        x_seq = self.x[t - self.seq_len + 1: t + 1]
        return torch.from_numpy(x_seq), t


# -------------------------
# Train / Predict
# -------------------------
def train_model_v7(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    device: str,
    fold_id: int,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_len = int(cfg["features"]["seq_len"])
    quantiles = [float(q) for q in cfg["model"]["quantiles"]]

    # pesos
    reg_alpha = float(cfg["train"].get("reg_alpha", 1.0))
    cls_alpha_side = float(cfg["train"].get("cls_alpha_side", 1.0))
    cls_alpha_hit = float(cfg["train"].get("cls_alpha_hit", 0.8))

    side_w = torch.tensor(cfg["train"].get("side_class_weights", [1.0, 2.5, 2.5]), dtype=torch.float32, device=device)
    hit_w = torch.tensor(cfg["train"].get("hit_class_weights", [2.0, 1.0, 2.0]), dtype=torch.float32, device=device)

    ce_side = torch.nn.CrossEntropyLoss(weight=side_w, reduction="none")
    ce_hit = torch.nn.CrossEntropyLoss(weight=hit_w, reduction="none")

    scaler = _make_scaler(train_df, feature_cols)

    ds_train = SeqDatasetV7(train_df, feature_cols, seq_len=seq_len)
    ds_val = SeqDatasetV7(val_df, feature_cols, seq_len=seq_len)

    def collate(batch):
        xs, y_side, ycL, ycS, yL, yS, sw = zip(*batch)
        x = torch.stack(xs, dim=0).numpy()
        x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype("float32")
        return (
            torch.from_numpy(x),
            torch.stack(y_side, dim=0),
            torch.stack(ycL, dim=0),
            torch.stack(ycS, dim=0),
            torch.stack(yL, dim=0),
            torch.stack(yS, dim=0),
            torch.stack(sw, dim=0),
        )

    dl_train = DataLoader(ds_train, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=True, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, drop_last=False, collate_fn=collate)

    mcfg = cfg["model"]
    model = ITransformerV7(
        lookback=seq_len,
        n_features=len(feature_cols),
        patch_len=int(mcfg.get("patch_len", 16)),
        d_model=int(mcfg.get("d_model", 192)),
        n_heads=int(mcfg.get("nhead", 6)),
        n_layers_time=int(mcfg.get("n_layers_time", 2)),
        n_layers_feat=int(mcfg.get("n_layers_feat", 4)),
        dropout=float(mcfg.get("dropout", 0.12)),
        n_quantiles=len(quantiles),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    def run_epoch(dl, train: bool):
        model.train(train)
        losses = []
        for X, y_side, ycL, ycS, yL, yS, sw in dl:
            X = X.to(device)
            y_side = y_side.to(device)
            ycL = ycL.to(device)
            ycS = ycS.to(device)
            yL = yL.to(device)
            yS = yS.to(device)
            sw = sw.to(device)

            out = model(X)
            side_logits = out["side_logits"]
            hitL_logits = out["hit_long_logits"]
            hitS_logits = out["hit_short_logits"]
            qL = out["q_long"]
            qS = out["q_short"]

            # REG
            loss_qL_ps = quantile_loss(qL, yL, quantiles, reduction="none")
            loss_qS_ps = quantile_loss(qS, yS, quantiles, reduction="none")
            loss_reg = (0.5 * (loss_qL_ps + loss_qS_ps) * sw).mean()

            # SIDE
            loss_side_ps = ce_side(side_logits, y_side)
            loss_side = (loss_side_ps * sw).mean()

            # HIT
            loss_hitL_ps = ce_hit(hitL_logits, ycL)
            loss_hitS_ps = ce_hit(hitS_logits, ycS)
            loss_hit = (0.5 * (loss_hitL_ps + loss_hitS_ps) * sw).mean()

            loss = reg_alpha * loss_reg + cls_alpha_side * loss_side + cls_alpha_hit * loss_hit

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
                opt.step()

            losses.append(float(loss.detach().cpu().item()))

        return float(np.mean(losses)) if losses else 0.0

    epochs = int(cfg["train"]["epochs"])
    best_val = float("inf")
    patience = int(cfg["train"].get("early_stop_patience", 10))
    bad = 0

    for ep in range(1, epochs + 1):
        tr = run_epoch(dl_train, True)
        va = run_epoch(dl_val, False)
        print(f"[v7 fold={fold_id}] ep={ep} loss_train={tr:.6f} loss_val={va:.6f}")

        if va < best_val - 1e-6:
            best_val = va
            bad = 0
            ckpt = out_dir / f"fold_{fold_id}_best.pt"
            torch.save({"model": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_std": scaler.std_}, ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"[v7 fold={fold_id}] early stop at ep={ep}")
                break

    ckpt = out_dir / f"fold_{fold_id}_best.pt"
    obj = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(obj["model"])
    scaler.mean_ = obj["scaler_mean"]
    scaler.std_ = obj["scaler_std"]
    model.to(device).eval()

    return model, scaler


def predict_v7(model, scaler, df, feature_cols, cfg, device: str) -> pd.DataFrame:
    model.eval()
    seq_len = int(cfg["features"]["seq_len"])
    quantiles = [float(q) for q in cfg["model"]["quantiles"]]

    ds = InferenceSeqDataset(df, feature_cols, seq_len=seq_len)

    def collate(batch):
        xs, ts = zip(*batch)
        x = torch.stack(xs, dim=0).numpy()
        x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype("float32")
        return torch.from_numpy(x), np.array(ts, dtype=np.int64)

    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, drop_last=False, collate_fn=collate)

    side_all, hitL_all, hitS_all, qL_all, qS_all, t_all = [], [], [], [], [], []

    with torch.no_grad():
        for x, ts in dl:
            x = x.to(device)
            out = model(x)

            side_all.append(torch.softmax(out["side_logits"], dim=1).cpu().numpy())
            hitL_all.append(torch.softmax(out["hit_long_logits"], dim=1).cpu().numpy())
            hitS_all.append(torch.softmax(out["hit_short_logits"], dim=1).cpu().numpy())
            qL_all.append(out["q_long"].cpu().numpy())
            qS_all.append(out["q_short"].cpu().numpy())
            t_all.append(ts)

    side_all = np.concatenate(side_all)
    hitL_all = np.concatenate(hitL_all)
    hitS_all = np.concatenate(hitS_all)
    qL_all = np.concatenate(qL_all)
    qS_all = np.concatenate(qS_all)
    t_all = np.concatenate(t_all)

    idx = df.index[t_all]
    out = pd.DataFrame(index=pd.DatetimeIndex(df.index[t_all]))

    out["p_flat"] = side_all[:, 0].astype("float32")
    out["p_long"] = side_all[:, 1].astype("float32")
    out["p_short"] = side_all[:, 2].astype("float32")

    out["pL_sl"] = hitL_all[:, 0].astype("float32")
    out["pL_time"] = hitL_all[:, 1].astype("float32")
    out["pL_tp"] = hitL_all[:, 2].astype("float32")

    out["pS_sl"] = hitS_all[:, 0].astype("float32")
    out["pS_time"] = hitS_all[:, 1].astype("float32")
    out["pS_tp"] = hitS_all[:, 2].astype("float32")

    for j, q in enumerate(quantiles):
        out[f"qL{int(q*100):02d}_reg"] = qL_all[:, j].astype("float32")
        out[f"qS{int(q*100):02d}_reg"] = qS_all[:, j].astype("float32")

    # desescalado a gross
    if bool(cfg["labels"].get("vol_scaled", True)) and "vol_scale" in df.columns:
        vs = df.loc[idx, "vol_scale"].astype("float32")
        out["vol_scale"] = vs.values
        for q in [10, 50, 90]:
            out[f"qL{q:02d}_gross"] = (out[f"qL{q:02d}_reg"] * out["vol_scale"]).astype("float32")
            out[f"qS{q:02d}_gross"] = (out[f"qS{q:02d}_reg"] * out["vol_scale"]).astype("float32")
        out["iqrL_gross"] = (out["qL90_gross"] - out["qL10_gross"]).abs().astype("float32")
        out["iqrS_gross"] = (out["qS90_gross"] - out["qS10_gross"]).abs().astype("float32")
    else:
        for q in [10, 50, 90]:
            out[f"qL{q:02d}_gross"] = out[f"qL{q:02d}_reg"].astype("float32")
            out[f"qS{q:02d}_gross"] = out[f"qS{q:02d}_reg"].astype("float32")
        out["iqrL_gross"] = (out["qL90_gross"] - out["qL10_gross"]).abs().astype("float32")
        out["iqrS_gross"] = (out["qS90_gross"] - out["qS10_gross"]).abs().astype("float32")

    # precios para ejecución
    out["open"] = df.loc[idx, "open"].astype("float32").values
    out["open_next"] = df["open"].shift(-1).loc[idx].astype("float32").values
    out["close"] = df.loc[idx, "close"].astype("float32").values
    out["high"] = df.loc[idx, "high"].astype("float32").values
    out["low"] = df.loc[idx, "low"].astype("float32").values
    out["atr"] = df.loc[idx, "atr"].astype("float32").values

    return out.sort_index()
