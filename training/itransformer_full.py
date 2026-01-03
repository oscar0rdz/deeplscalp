import math
import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.
    Normalize per-sample per-feature over time, then de-normalize if needed.
    """
    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, n_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, n_features))

    def forward(self, x, mode: str):
        # x: [B, L, F]
        if mode == "norm":
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp_min(self.eps)
            x_n = (x - mu) / sd
            if self.affine:
                x_n = x_n * self.gamma + self.beta
            return x_n, mu, sd
        elif mode == "denorm":
            x_n, mu, sd = x
            if self.affine:
                x_n = (x_n - self.beta) / (self.gamma.clamp_min(self.eps))
            return x_n * sd + mu
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

class ITransformerV6(nn.Module):
    """
    V6 Multi-Head iTransformer:
      - RevIN
      - Patch embedding over time
      - Temporal encoder shared across variates (per-feature dynamics)
      - Feature encoder across variates (iTransformer core)
      - Multi-task heads: SIDE (3) + HIT_L (3) + HIT_S (3) + Q_L (quantiles) + Q_S (quantiles)
      - Returns embedding for OOD scoring
    """
    def __init__(
        self,
        lookback: int,
        n_features: int,
        patch_len: int = 16,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers_time: int = 2,
        n_layers_feat: int = 4,
        dropout: float = 0.12,
        n_quantiles: int = 3,
    ):
        super().__init__()
        self.lookback = lookback
        self.n_features = n_features
        self.patch_len = patch_len
        self.d_model = d_model

        self.revin = RevIN(n_features=n_features, affine=True)

        # Patch projection: each feature becomes a sequence of patches
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Shared temporal encoder over patches (applied per feature)
        time_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(time_layer, num_layers=n_layers_time)

        # Feature embeddings
        self.feat_emb = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.normal_(self.feat_emb, mean=0.0, std=0.02)

        # Cross-feature encoder (variates as tokens)
        feat_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.feature_encoder = nn.TransformerEncoder(feat_layer, num_layers=n_layers_feat)

        # Attention pooling over features (learnable query)
        self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_q, mean=0.0, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Heads V6
        self.side_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),  # flat, long, short
        )
        self.hitL_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),  # sl, time, tp
        )
        self.hitS_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),  # sl, time, tp
        )
        self.quantL_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_quantiles),
        )
        self.quantS_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_quantiles),
        )

    def _to_patches(self, x):
        # x: [B, L, F]
        B, L, F = x.shape
        P = self.patch_len
        # pad to multiple of P
        pad = (P - (L % P)) % P
        if pad:
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
            L = L + pad
        N = L // P
        # reshape into patches
        x = x.view(B, N, P, F)           # [B, N, P, F]
        x = x.permute(0, 3, 1, 2)        # [B, F, N, P]
        return x, N

    def forward(self, x):
        # x: [B, L, F]
        x, mu, sd = self.revin(x, mode="norm")

        # Patching
        xpf, N = self._to_patches(x)     # [B, F, N, P]
        B, F, N, P = xpf.shape

        # Project patches -> embeddings
        z = self.patch_proj(xpf)         # [B, F, N, d_model]
        z = z.reshape(B*F, N, self.d_model)
        z = self.temporal_encoder(z)     # [B*F, N, d_model]
        z = z.mean(dim=1)                # [B*F, d_model] pool patches
        z = z.view(B, F, self.d_model)   # [B, F, d_model]

        # Add feature embedding + cross-feature encoder
        z = z + self.feat_emb
        z = self.feature_encoder(z)      # [B, F, d_model]

        # Attention pooling
        q = self.pool_q.expand(B, -1, -1)               # [B, 1, d_model]
        pooled, _ = self.pool_attn(q, z, z)             # [B, 1, d_model]
        emb = pooled.squeeze(1)                          # [B, d_model]

        side_logits = self.side_head(emb)               # [B, 3]
        hitL_logits = self.hitL_head(emb)               # [B, 3]
        hitS_logits = self.hitS_head(emb)               # [B, 3]
        qL = self.quantL_head(emb)                       # [B, Q]
        qS = self.quantS_head(emb)                       # [B, Q]
        return side_logits, hitL_logits, hitS_logits, qL, qS, emb
