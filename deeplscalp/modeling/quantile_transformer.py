from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    nhead: int
    num_layers: int
    dropout: float
    quantiles: List[float]
    norm_first: bool


class QuantileTransformer(nn.Module):
    """
    Transformer encoder para regresión cuantílica:
    entrada: (B, T, F)
    salida: (B, Q) quantiles de y
    """
    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.n_features = n_features
        self.q = cfg.quantiles

        self.in_proj = nn.Linear(n_features, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=cfg.norm_first,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, len(self.q)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = self.encoder(z)
        # pooling: último token (estable en series)
        h = z[:, -1, :]
        out = self.head(h)
        return out


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: List[float],
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Pinball loss para regresión cuantílica.

    pred:   (B, Q)
    target: (B,) o (B,1)

    reduction="none" -> (B,)
    reduction="mean" -> escalar
    """
    if target.ndim == 2 and target.shape[1] == 1:
        y = target
    else:
        y = target.unsqueeze(1)  # (B,1)

    qs = quantiles
    losses = []
    for i, q in enumerate(qs):
        e = y - pred[:, i:i+1]  # (B,1)
        losses.append(torch.maximum((q - 1) * e, q * e))  # (B,1)

    per_sample = torch.sum(torch.cat(losses, dim=1), dim=1)  # (B,)

    if reduction == "none":
        return per_sample
    return torch.mean(per_sample)
