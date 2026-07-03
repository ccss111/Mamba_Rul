import torch
from torch import nn

from model.GAT import SensorSpatialGAT
from model.LSTM_Attention import Seq2SeqDecoder, Seq2SeqEncoder

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - import error is environment-specific
    raise ImportError(
        "mamba_ssm is required for the Mamba-based model defined in this file."
    ) from exc


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class ResidualGatedMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1, d_conv=4, expand=2):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.gate = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = RMSNorm(d_model)

    def forward(self, x):
        residual = x
        x_norm = self.pre_norm(x)
        if not x_norm.is_cuda:
            raise RuntimeError("ResidualGatedMambaBlock requires CUDA tensors; CPU fallback is disabled.")
        mamba_out = self.mamba(x_norm)
        mamba_out = self.dropout(mamba_out)
        gate = torch.sigmoid(self.gate(torch.cat([x_norm, mamba_out], dim=-1)))
        x = residual + gate * mamba_out
        return self.post_norm(x)


class MultiScaleGATBranch(nn.Module):
    def __init__(
        self,
        num_sensors,
        branch_dim,
        gat_hidden_dim=16,
        gat_num_layers=2,
        gat_embed_dim=16,
        gat_topks=(3, 5, 7),
        graph_mode="dynamic_knn",
        gat_dropout=0.0,
        gat_alpha=0.1,
    ):
        super().__init__()
        if not gat_topks:
            raise ValueError("gat_topks cannot be empty")
        self.branch_dim = branch_dim
        self.gat_modules = nn.ModuleList(
            [
                SensorSpatialGAT(
                    num_sensors=num_sensors,
                    in_features=1,
                    hidden_features=gat_hidden_dim,
                    out_features=branch_dim,
                    num_layers=gat_num_layers,
                    embed_dim=gat_embed_dim,
                    topk=topk,
                    graph_mode=graph_mode,
                    dropout=gat_dropout,
                    alpha=gat_alpha,
                )
                for topk in gat_topks
            ]
        )
        self.concat_linear = nn.Linear(branch_dim * len(gat_topks), branch_dim)
        self.concat_norm = RMSNorm(branch_dim)
        self.branch_head = nn.Linear(branch_dim, 1)

    def forward(self, x):
        branch_outputs = [gat(x) for gat in self.gat_modules]
        fused = torch.cat(branch_outputs, dim=-1)
        fused = self.concat_linear(fused)
        fused = self.concat_norm(fused)
        fused = torch.sigmoid(fused)
        pooled = fused.mean(dim=1)
        pred = self.branch_head(pooled)
        return pred, pooled


class AEFLSTMBranch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, attention_size):
        super().__init__()
        self.encoder = Seq2SeqEncoder(
            input_size=input_size,
            num_layers=num_layers,
            num_hidden=hidden_size,
        )
        self.decoder = Seq2SeqDecoder(
            input_size=input_size,
            num_layers=num_layers,
            num_hidden=hidden_size,
            seq_len=seq_len,
            attention_size=attention_size,
            use_aef=True,
        )

    def forward(self, x):
        encoder_output, encoder_state = self.encoder(x)
        decoder_input, attention_weights = self.decoder.attention(encoder_output, encoder_state[0])
        decoder_output, _ = self.decoder.lstm(decoder_input, encoder_state)
        hidden = decoder_output[:, -1, :]
        pred = self.decoder.Linear(hidden)
        return pred, hidden, attention_weights


class MambaED3KGATRegressor(nn.Module):
    def __init__(
        self,
        input_size=14,
        sequence_len=30,
        lstm_hidden_dim=32,
        lstm_num_layers=2,
        attention_size=32,
        mamba_num_layers=2,
        mamba_d_state=16,
        mamba_dropout=0.1,
        mamba_d_conv=4,
        mamba_expand=2,
        use_spatial_gat=True,
        graph_mode="dynamic_knn",
        gat_hidden_dim=16,
        gat_num_layers=2,
        gat_embed_dim=16,
        gat_dropout=0.0,
        gat_alpha=0.1,
        gat_topks=(3, 5, 7),
        context_dropout=0.1,
        fusion_dropout=0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.sequence_len = sequence_len
        self.use_spatial_gat = bool(use_spatial_gat)

        self.mamba_blocks = nn.ModuleList(
            [
                ResidualGatedMambaBlock(
                    d_model=input_size,
                    d_state=mamba_d_state,
                    dropout=mamba_dropout,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                )
                for _ in range(int(mamba_num_layers))
            ]
        )
        self.context_projection = nn.Linear(input_size, lstm_hidden_dim)
        self.context_projection_norm = RMSNorm(lstm_hidden_dim)

        if self.use_spatial_gat:
            self.gat_branch = MultiScaleGATBranch(
                num_sensors=input_size,
                branch_dim=lstm_hidden_dim,
                gat_hidden_dim=gat_hidden_dim,
                gat_num_layers=gat_num_layers,
                gat_embed_dim=gat_embed_dim,
                gat_topks=gat_topks,
                graph_mode=graph_mode,
                gat_dropout=gat_dropout,
                gat_alpha=gat_alpha,
            )
        else:
            self.gat_branch = None
            self.gat_fallback_projection = nn.Sequential(
                nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
                RMSNorm(lstm_hidden_dim),
                nn.Sigmoid(),
            )
            self.gat_fallback_head = nn.Linear(lstm_hidden_dim, 1)

        self.lstm_branch = AEFLSTMBranch(
            input_size=input_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            seq_len=sequence_len,
            attention_size=attention_size,
        )

        gate_input_dim = lstm_hidden_dim * 3 + 2
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_input_dim, lstm_hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = RMSNorm(lstm_hidden_dim)
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.output_head = nn.Linear(lstm_hidden_dim, 1)

    def _run_mamba_context(self, x):
        context = x
        for block in self.mamba_blocks:
            context = block(context)
        return context

    def forward(self, encoder_x):
        context_seq = self._run_mamba_context(encoder_x)
        raw_summary = context_seq.mean(dim=1)
        context_summary = self.context_projection(raw_summary)
        context_summary = self.context_projection_norm(context_summary)

        if self.use_spatial_gat:
            gat_pred, gat_repr = self.gat_branch(context_seq)
        else:
            gat_repr = self.gat_fallback_projection(raw_summary)
            gat_pred = self.gat_fallback_head(gat_repr)

        lstm_pred, lstm_repr, attention_weights = self.lstm_branch(context_seq)

        gate_input = torch.cat(
            [
                gat_repr,
                lstm_repr,
                context_summary,
                gat_pred,
                lstm_pred,
            ],
            dim=-1,
        )
        gate = self.fusion_gate(gate_input)
        fused = gate * gat_repr + (1.0 - gate) * lstm_repr
        fused = fused + context_summary
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)
        output = self.output_head(fused)
        return output, attention_weights



__all__ = [
    "RMSNorm",
    "ResidualGatedMambaBlock",
    "MultiScaleGATBranch",
    "AEFLSTMBranch",
    "MambaED3KGATRegressor",
]
