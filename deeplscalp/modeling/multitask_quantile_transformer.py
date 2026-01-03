from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    nhead: int
    num_layers: int
    dropout: float
    quantiles: list[float]
    norm_first: bool
    cls_weight: float = 0.5   # peso BCE
    reg_weight: float = 0.5   # peso pinball

class MultiTaskQuantileTransformer(nn.Module):
    """
    V5: Salidas para TBM 3-clases
      - q_pred: (B, Q) cuantiles de y_reg
      - cls_logits: (B, 3) logits de [SL, TIME, TP]
    """
    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.q = cfg.quantiles
        self.cfg = cfg

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

        self.shared = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.head_q = nn.Linear(cfg.d_model, len(self.q))
        self.head_cls = nn.Linear(cfg.d_model, 3)  # 3 clases: SL=0, TIME=1, TP=2

    def forward(self, x: torch.Tensor):
        z = self.in_proj(x)
        z = self.encoder(z)
        h = z[:, -1, :]
        h = self.shared(h)
        q_pred = self.head_q(h)
        cls_logits = self.head_cls(h)  # (B, 3)
        return q_pred, cls_logits

def quantile_loss(pred_q: torch.Tensor, target: torch.Tensor, quantiles: list[float], w=None, reduction="mean") -> torch.Tensor:
    y = target.unsqueeze(1)
    losses = []
    for i, q in enumerate(quantiles):
        e = y - pred_q[:, i:i+1]
        losses.append(torch.maximum((q - 1.0) * e, q * e))
    per_sample = torch.cat(losses, dim=1).sum(dim=1)

    if reduction == "none":
        return per_sample
    if w is None:
        return per_sample.mean()
    w = w.to(per_sample.dtype).clamp(min=0.0)
    denom = w.sum().clamp(min=1e-12)
    return (per_sample * w).sum() / denom

def multitask_loss(q_pred, up_logit, y_reg, y_up, cfg: ModelConfig, w=None):
    # regresión
    l_reg = quantile_loss(q_pred, y_reg, cfg.quantiles, w=w)

    # clasificación
    y_up = y_up.to(up_logit.dtype)
    if w is None:
        l_cls = F.binary_cross_entropy_with_logits(up_logit, y_up)
    else:
        w2 = w.to(up_logit.dtype).clamp(min=0.0)
        denom = w2.sum().clamp(min=1e-12)
        l_cls = (F.binary_cross_entropy_with_logits(up_logit, y_up, reduction="none") * w2).sum() / denom

    return cfg.reg_weight * l_reg + cfg.cls_weight * l_cls, {"l_reg": float(l_reg.detach().cpu()), "l_cls": float(l_cls.detach().cpu())}
