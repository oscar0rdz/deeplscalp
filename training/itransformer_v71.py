from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantile_loss(pred_q: torch.Tensor, y: torch.Tensor, quantiles: list[float], reduction="mean"):
    # pred_q: (B, Q), y: (B,)
    y = y.view(-1, 1)
    qs = torch.tensor(quantiles, device=pred_q.device, dtype=pred_q.dtype).view(1, -1)
    e = y - pred_q
    loss = torch.maximum(qs * e, (qs - 1.0) * e)  # pinball
    loss = loss.sum(dim=1)  # (B,)
    if reduction == "none":
        return loss
    return loss.mean()


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor):
        # x: (B,T,F) -> normalize per sample, per feature
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        xn = (x - mean) / std
        xn = xn * self.gamma + self.beta
        return xn, mean, std

    def denorm(self, xn: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        x = (xn - self.beta) / (self.gamma + 1e-12)
        return x * std + mean


@dataclass
class ITransV71Config:
    seq_len: int
    n_features: int
    d_model: int = 192
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.10
    quantiles: tuple = (0.10, 0.50, 0.90)
    norm_first: bool = True


class ITransformerV71(nn.Module):
    """
    iTransformer-like:
      - Tokens = variables/features
      - Cada token recibe el histórico temporal completo (T) proyectado a d_model.
      - TransformerEncoder atiende relaciones cross-variable.
      - Pool global -> heads.

    Heads:
      side:    3 (flat/long/short)
      hitL:    3 (sl/time/tp)
      hitS:    3 (sl/time/tp)
      qL:      Q
      qS:      Q
      regime:  4 (range/trend_up/trend_down/spike)
      event:   4 (none/breakout/rebound/spike)
    """
    def __init__(self, cfg: ITransV71Config):
        super().__init__()
        self.cfg = cfg
        self.revin = RevIN(cfg.n_features)

        # Proyección: para cada feature token, proyecta historia T -> d_model
        self.proj = nn.Linear(cfg.seq_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=cfg.norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)

        Q = len(cfg.quantiles)

        # heads
        self.side_head = nn.Linear(cfg.d_model, 3)
        self.hitL_head = nn.Linear(cfg.d_model, 3)
        self.hitS_head = nn.Linear(cfg.d_model, 3)
        self.regime_head = nn.Linear(cfg.d_model, 4)
        self.event_head = nn.Linear(cfg.d_model, 4)

        self.qL_head = nn.Linear(cfg.d_model, Q)
        self.qS_head = nn.Linear(cfg.d_model, Q)

    def forward(self, x: torch.Tensor):
        # x: (B,T,F)
        B, T, F_ = x.shape
        assert T == self.cfg.seq_len, f"seq_len mismatch: got {T}, expected {self.cfg.seq_len}"
        assert F_ == self.cfg.n_features, f"feature mismatch: got {F_}, expected {self.cfg.n_features}"

        x, mean, std = self.revin(x)

        # (B,T,F) -> (B,F,T)
        xv = x.transpose(1, 2)

        # proyecta cada token-variable: (B,F,T) -> (B,F,d_model)
        tok = self.proj(xv)

        # encoder cross-variable: tokens=F
        z = self.encoder(tok)
        z = self.norm(self.dropout(z))

        # pool global (promedio de tokens)
        emb = z.mean(dim=1)

        side_logits = self.side_head(emb)
        hitL_logits = self.hitL_head(emb)
        hitS_logits = self.hitS_head(emb)
        reg_logits = self.regime_head(emb)
        evt_logits = self.event_head(emb)
        qL = self.qL_head(emb)
        qS = self.qS_head(emb)

        return side_logits, hitL_logits, hitS_logits, qL, qS, reg_logits, evt_logits, emb
