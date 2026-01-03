# training/itransformer_v7.py
import torch
import torch.nn as nn

EPS = 1e-6


class RevIN(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, n_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, n_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp_min(self.eps)
            x_n = (x - mu) / sd
            if self.affine:
                x_n = x_n * self.gamma + self.beta
            return x_n, mu, sd
        else:
            raise ValueError("RevIN solo se usa en modo 'norm' en V7.")


class ITransformerV7(nn.Module):
    """
    iTransformer-like multitarea:
      - patching temporal por feature
      - encoder temporal compartido
      - encoder cross-feature
      - pooling por atención
      - inyección de contexto no estacionario (mu, log(sd))
      - heads:
          side (FLAT/LONG/SHORT)
          hit_long (SL/TIME/TP)
          hit_short (SL/TIME/TP)
          q_long (quantiles)
          q_short (quantiles)
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

        self.patch_proj = nn.Linear(patch_len, d_model)

        time_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(time_layer, num_layers=n_layers_time)

        self.feat_emb = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.normal_(self.feat_emb, mean=0.0, std=0.02)

        feat_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.feature_encoder = nn.TransformerEncoder(feat_layer, num_layers=n_layers_feat)

        self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_q, mean=0.0, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # contexto no estacionario
        self.ctx_mlp = nn.Sequential(
            nn.Linear(2 * n_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # heads
        def head(out_dim: int):
            return nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, out_dim),
            )

        self.head_side = head(3)
        self.head_hit_long = head(3)
        self.head_hit_short = head(3)
        self.head_q_long = head(n_quantiles)
        self.head_q_short = head(n_quantiles)

    def _to_patches(self, x):
        # x: [B,L,F]
        B, L, F = x.shape
        P = self.patch_len
        pad = (P - (L % P)) % P
        if pad:
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
            L = L + pad
        N = L // P
        x = x.view(B, N, P, F).permute(0, 3, 1, 2)  # [B,F,N,P]
        return x, N

    def forward(self, x):
        # x: [B,L,F]
        x_n, mu, sd = self.revin(x, mode="norm")  # mu/sd: [B,1,F]

        xpf, N = self._to_patches(x_n)
        B, F, N, P = xpf.shape

        z = self.patch_proj(xpf)                  # [B,F,N,d]
        z = z.reshape(B * F, N, self.d_model)
        z = self.temporal_encoder(z).mean(dim=1)  # [B*F,d]
        z = z.view(B, F, self.d_model)

        z = z + self.feat_emb
        z = self.feature_encoder(z)

        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, z, z)
        emb = pooled.squeeze(1)  # [B,d]

        mu_f = mu.squeeze(1)
        sd_f = sd.squeeze(1).clamp_min(EPS)
        ctx_in = torch.cat([mu_f, torch.log(sd_f)], dim=1)
        emb = emb + self.ctx_mlp(ctx_in)

        return {
            "side_logits": self.head_side(emb),
            "hit_long_logits": self.head_hit_long(emb),
            "hit_short_logits": self.head_hit_short(emb),
            "q_long": self.head_q_long(emb),
            "q_short": self.head_q_short(emb),
            "emb": emb,
        }
