import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSGFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(round(d_model * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, S, D]
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        x = residual + self.drop1(attn_out)

        x = x + self.mlp(self.norm2(x))
        # Note: older PyTorch versions return attention averaged across heads: [B, S, S].
        return x, attn_weights


class MambaSSMBlock(nn.Module):
    """A small, pure-PyTorch selective state space block.

    This is *not* an exact drop-in reproduction of the official mamba-ssm implementation.
    It provides the core idea: input-dependent discretization (delta) and per-channel
    diagonal SSM state update, without any encoder-decoder structure.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)

        self.in_norm = nn.LayerNorm(d_model)

        # Projections to parameterize the selective SSM.
        self.in_proj = nn.Linear(d_model, d_model)
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_model * d_state)
        self.C_proj = nn.Linear(d_model, d_model * d_state)

        # Diagonal A (negative for stability). Shape: [D, N]
        self.A_log = nn.Parameter(torch.zeros(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {D}.")

        residual = x
        x = self.in_norm(x)

        u = self.in_proj(x)  # [B, T, D]
        delta = F.softplus(self.delta_proj(x))  # [B, T, D]

        B_t = self.B_proj(x).view(B, T, D, self.d_state)  # [B, T, D, N]
        C_t = self.C_proj(x).view(B, T, D, self.d_state)  # [B, T, D, N]

        A = -torch.exp(self.A_log)  # [D, N]

        # State per batch+channel.
        state = x.new_zeros((B, D, self.d_state))
        ys = []
        for t in range(T):
            dt = delta[:, t, :].unsqueeze(-1)  # [B, D, 1]
            # Discretized diagonal update: state = exp(A*dt)*state + B*dt*u
            state = state * torch.exp(A.unsqueeze(0) * dt) + B_t[:, t, :, :] * (dt * u[:, t, :].unsqueeze(-1))
            y = (state * C_t[:, t, :, :]).sum(dim=-1) + self.D.unsqueeze(0) * u[:, t, :]
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [B, T, D]
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class SGFormerMambaRegressor(nn.Module):
    """SGFormer (spatial across sensors) + Mamba-style SSM (temporal) regressor.

    Input:
        x: [B, T, S]  (T=sequence_len, S=feature_num / sensors)
    Output:
        pred: [B, 1] normalized RUL in [0, 1]
        attn: optional spatial attention weights for the last SGFormer layer
    """

    def __init__(
        self,
        num_sensors: int,
        d_model: int = 64,
        spatial_num_layers: int = 2,
        spatial_num_heads: int = 4,
        spatial_dropout: float = 0.0,
        temporal_num_layers: int = 2,
        temporal_d_state: int = 16,
        temporal_dropout: float = 0.0,
        pooling: str = "mean",
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.d_model = int(d_model)
        self.pooling = str(pooling)

        # Per-sensor scalar -> token embedding
        self.sensor_in = nn.Linear(1, d_model)

        self.spatial_blocks = nn.ModuleList(
            [
                SpatialSGFormerBlock(
                    d_model=d_model,
                    num_heads=spatial_num_heads,
                    dropout=spatial_dropout,
                )
                for _ in range(int(spatial_num_layers))
            ]
        )

        self.temporal_blocks = nn.ModuleList(
            [
                MambaSSMBlock(
                    d_model=d_model,
                    d_state=temporal_d_state,
                    dropout=temporal_dropout,
                )
                for _ in range(int(temporal_num_layers))
            ]
        )

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        # x: [B, T, S]
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, T, S], got {tuple(x.shape)}")
        B, T, S = x.shape
        if S != self.num_sensors:
            raise ValueError(f"Expected num_sensors={self.num_sensors}, got S={S}")

        # Spatial encoding per timestep: reshape to (B*T, S, 1)
        x_tokens = x.reshape(B * T, S, 1)
        x_tokens = self.sensor_in(x_tokens)  # [B*T, S, D]

        attn_last = None
        for block in self.spatial_blocks:
            x_tokens, attn_last = block(x_tokens)

        # Pool sensors -> timestep embedding
        if self.pooling == "mean":
            x_time = x_tokens.mean(dim=1)  # [B*T, D]
        elif self.pooling == "cls":
            x_time = x_tokens[:, 0, :]  # [B*T, D]
        else:
            raise ValueError("pooling must be one of ['mean', 'cls']")

        x_time = x_time.view(B, T, self.d_model)  # [B, T, D]

        # Temporal modeling
        for block in self.temporal_blocks:
            x_time = block(x_time)

        # Predict from last step
        x_last = x_time[:, -1, :]
        x_last = self.head_norm(x_last)
        pred = self.head(x_last)
        pred = torch.sigmoid(pred)

        # Match the rest of the codebase's convention: return (pred, attn)
        return pred, attn_last
