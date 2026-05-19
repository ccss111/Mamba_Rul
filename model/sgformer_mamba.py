import torch
import torch.nn as nn

from mamba_ssm import Mamba


def _sgformer_full_attention_conv(
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    output_attn: bool = False,
    eps: float = 1e-6,
):
    """Batched variant of `full_attention_conv` from `original_SGFormer.py`.

    Shapes:
        qs: [B, N, H, M]
        ks: [B, N, H, M]
        vs: [B, N, H, D]

    Returns:
        attn_output: [B, N, H, D]
        attention (optional): [B, N, N]

    Notes:
        This is *not* softmax attention. It matches SGFormer codepath that uses
        a normalization term with an added constant N.
    """

    if qs.dim() != 4 or ks.dim() != 4 or vs.dim() != 4:
        raise ValueError(
            "Expected qs/ks/vs with shapes [B, N, H, *], "
            f"got qs={tuple(qs.shape)}, ks={tuple(ks.shape)}, vs={tuple(vs.shape)}"
        )

    B, N, H, _ = qs.shape
    if ks.shape[:3] != (B, N, H) or vs.shape[:3] != (B, N, H):
        raise ValueError(
            "qs/ks/vs must share the same [B, N, H] dims, "
            f"got qs={tuple(qs.shape)}, ks={tuple(ks.shape)}, vs={tuple(vs.shape)}"
        )

    # normalize input (per-batch, to avoid cross-batch coupling)
    qs = qs / (qs.norm(p=2, dim=(1, 2, 3), keepdim=True) + eps)
    ks = ks / (ks.norm(p=2, dim=(1, 2, 3), keepdim=True) + eps)

    # numerator
    kvs = torch.einsum("bnhm,bnhd->bhmd", ks, vs)  # [B, H, M, D]
    attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [B, N, H, D]
    attention_num = attention_num + (N * vs)

    # denominator
    all_ones = ks.new_ones((N,))
    ks_sum = torch.einsum("bnhm,n->bhm", ks, all_ones)  # [B, H, M]
    attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [B, N, H]

    attention_normalizer = attention_normalizer.unsqueeze(-1)  # [B, N, H, 1]
    attention_normalizer = attention_normalizer + float(N)

    attn_output = attention_num / (attention_normalizer + eps)

    if not output_attn:
        return attn_output

    attention = torch.einsum("bnhm,blhm->bnlh", qs, ks).mean(dim=-1)  # [B, N, N]
    normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdim=True)  # [B, N, 1]
    attention = attention / (normalizer + eps)
    return attn_output, attention


class SpatialTransConvLayer(nn.Module):
    """SGFormer-style attention layer (from `TransConvLayer`) for spatial tokens.

    This follows `original_SGFormer.py` behavior:
    - projects to (out_channels * num_heads)
    - reshapes into [B, N, H, out_channels]
    - runs `full_attention_conv`
    - averages over heads
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        use_weight: bool = True,
    ):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads) if use_weight else None

        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)
        self.use_weight = bool(use_weight)

    def forward(self, x: torch.Tensor, output_attn: bool = False):
        # x: [B, N, C]
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, N, C], got {tuple(x.shape)}")

        query = self.Wq(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.out_channels)
        key = self.Wk(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.out_channels)
        else:
            value = x.reshape(x.shape[0], x.shape[1], 1, self.out_channels)

        if output_attn:
            attention_output, attn = _sgformer_full_attention_conv(query, key, value, output_attn=True)
        else:
            attention_output = _sgformer_full_attention_conv(query, key, value, output_attn=False)

        final_output = attention_output.mean(dim=2)  # mean over heads -> [B, N, D]
        if output_attn:
            return final_output, attn
        return final_output


class SpatialSGFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        alpha: float = 0.5,
        use_residual: bool = True,
        use_act: bool = True,
    ):
        super().__init__()
        # NOTE: `mlp_ratio` is kept only for backward compatibility with the previous
        # signature. SGFormer (as implemented in `original_SGFormer.py`) does not use
        # a Transformer FFN here.
        _ = mlp_ratio

        self.alpha = float(alpha)
        self.use_residual = bool(use_residual)
        self.use_act = bool(use_act)

        self.conv = SpatialTransConvLayer(
            in_channels=d_model,
            out_channels=d_model,
            num_heads=num_heads,
            use_weight=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, S, D]
        residual = x
        x_out, attn = self.conv(x, output_attn=True)

        if self.use_residual:
            x = self.alpha * x_out + (1.0 - self.alpha) * residual
        else:
            x = x_out

        x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        x = self.drop(x)

        return x, attn


class TemporalMambaBlock(nn.Module):
    """Temporal block using official mamba-ssm `Mamba`.

    Keeps a Transformer-like pre-norm residual structure:
        x = x + Dropout(Mamba(LayerNorm(x)))

    Input/Output: [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dropout: float = 0.0,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if not x.is_cuda:
            raise RuntimeError(
                "mamba_ssm.Mamba requires CUDA tensors. "
                "Please move inputs/model to CUDA (and do not use --no-cuda)."
            )

        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


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
                TemporalMambaBlock(
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
