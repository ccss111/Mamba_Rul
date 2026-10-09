from .GAT import SensorSpatialGAT
from mamba_ssm import Mamba

import torch
from torch import nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 mamba_num_layers,
                 mamba_d_model,
                 mamba_d_state=16,
                 mamba_dropout=0.1,
                 mamba_d_conv=4,
                 mamba_expand=2):
        super(Seq2SeqEncoder, self).__init__()
        self.input_size = int(input_size)
        self.mamba_d_model = int(mamba_d_model)
        self.mamba_num_layers = int(mamba_num_layers)
        self.output_size = self.mamba_d_model
        self.hidden_size = self.mamba_d_model
        self.input_projection = None
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.mamba_d_model,
                d_state=mamba_d_state,
                dropout=mamba_dropout,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
            for _ in range(self.mamba_num_layers)
        ])

    def forward(self, x):
        if self.input_projection is None:
            input_size = int(x.size(-1))
            self.input_projection = (
                nn.Identity()
                if input_size == self.mamba_d_model
                else nn.Linear(input_size, self.mamba_d_model)
            ).to(x.device)
        x = self.input_projection(x)
        output = x
        for block in self.blocks:
            output = block(output)
        hidden_state = output[:, -1, :].unsqueeze(0).repeat(self.mamba_num_layers, 1, 1)
        cell_state = torch.zeros_like(hidden_state)
        return output, (hidden_state, cell_state)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 两层因果卷积提取局部退化模式，残差支路保留原始时序信息。
        residual = self.residual(x)
        output = self.conv1(x)
        output = self.dropout(self.activation(self.norm1(output)))
        output = self.conv2(output)
        output = self.dropout(self.activation(self.norm2(output)))
        return self.activation(output + residual)


class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale causal TCN followed by temporal and scale-wise fusion."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 4),
        dropout=0.1,
    ):
        super().__init__()
        # 每个 kernel/dilation 组合代表一个时间感受野，数量必须一一对应。
        if len(kernel_sizes) != len(dilations) or not kernel_sizes:
            raise ValueError("kernel_sizes and dilations must have the same non-zero length")
        # 各尺度独立提取特征，之后再由注意力模块自适应决定尺度贡献。
        self.scales = nn.ModuleList([
            TCNResidualBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout,
            )
            for kernel_size, dilation in zip(kernel_sizes, dilations)
        ])
        # 每个尺度生成 (B, 1, T) 的时间权重，用于突出当前退化阶段的重要时间步。
        self.temporal_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(out_channels, 1, kernel_size=1),
                nn.Softmax(dim=-1),
            )
            for _ in self.scales
        ])
        # 将每个尺度的时间加权摘要 (B, C) 映射为一个尺度分数，随后在尺度维归一化。
        self.scale_attention = nn.Sequential(
            nn.Linear(out_channels, max(1, out_channels // 2)),
            nn.ReLU(),
            nn.Linear(max(1, out_channels // 2), 1),
        )
        self.output_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # 输入来自 GAT/Mamba，形状为 (B, T, C)；Conv1d 需要 (B, C, T)。
        x = x.transpose(1, 2)
        scale_outputs = [scale(x) for scale in self.scales]
        # 每个尺度的输出保持为 (B, C, T)，时间注意力沿 T 维计算。
        temporal_weights = [attention(output) for output, attention in zip(
            scale_outputs, self.temporal_attention
        )]

        # 时间加权摘要只用于决定尺度重要性，结果形状为 (B, S, C)。
        scale_descriptors = torch.stack([
            (output * weights).sum(dim=-1)
            for output, weights in zip(scale_outputs, temporal_weights)
        ], dim=1)
        scale_logits = self.scale_attention(scale_descriptors).squeeze(-1)
        # S 为尺度数；softmax 后每个样本的所有尺度权重之和为 1。
        scale_weights = torch.softmax(scale_logits, dim=1)
        # 将 (B, S, 1) 的尺度权重广播到每个通道和时间步，保留完整序列信息。
        fused = sum(
            output * scale_weights[:, index:index + 1, None]
            for index, output in enumerate(scale_outputs)
        )
        # 恢复为后续 Mamba/解码器使用的 (B, T, C) 布局。
        fused = fused.transpose(1, 2)
        return self.output_norm(fused)


class Seq2SeqDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 mamba_num_layers,
                 mamba_d_model,
                 use_aef=False,
                 mamba_d_state=16,
                 mamba_dropout=0.1,
                 mamba_d_conv=4,
                 mamba_expand=2):
        super(Seq2SeqDecoder, self).__init__()
        self.input_size = int(input_size)
        self.mamba_d_model = int(mamba_d_model)
        self.mamba_num_layers = int(mamba_num_layers)
        self.use_aef = bool(use_aef)
        self.input_projection = (
            nn.Identity()
            if self.input_size == self.mamba_d_model
            else nn.Linear(self.input_size, self.mamba_d_model)
        )
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=mamba_d_model,
                d_state=mamba_d_state,
                dropout=mamba_dropout,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
            for _ in range(self.mamba_num_layers)
        ])

    def forward(self, encoder_output, encoder_state):
        decoder_input = self.input_projection(encoder_output)
        output = decoder_input
        for block in self.blocks:
            output = block(output)
        output = output[:, -1, :]
        return output, None


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 use_spatial_gat=True,
                 graph_mode='dynamic_knn',
                 num_sensors=14,
                 gat_hidden_dim=8,
                 gat_out_dim=16,
                 gat_num_layers=2,
                 gat_embed_dim=16,
                 gat_topk=5,
                 dropout=0.1,
                 gat_alpha=0.1,
                 tcn_channels=16,
                 tcn_kernel_sizes=(3, 5, 7),
                 tcn_dilations=(1, 2, 4),
                 use_decoder=True):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_spatial_gat = use_spatial_gat
        self.graph_mode = graph_mode
        self.use_decoder = bool(use_decoder)
        self.tcn_channels = int(tcn_channels)

        if self.use_decoder and self.decoder is None:
            raise ValueError("decoder cannot be None when use_decoder=True")

        if use_spatial_gat:
            self.spatial_gat = SensorSpatialGAT(
                num_sensors=num_sensors,
                in_features=1,
                hidden_features=gat_hidden_dim,
                out_features=gat_out_dim,
                num_layers=gat_num_layers,
                embed_dim=gat_embed_dim,
                topk=gat_topk,
                graph_mode=graph_mode,
                dropout=dropout,
                alpha=gat_alpha,
            )
        else:
            self.spatial_gat = None

        tcn_input_size = gat_out_dim if use_spatial_gat else num_sensors
        self.tcn_fusion = MultiScaleFeatureFusion(
            in_channels=tcn_input_size,
            out_channels=self.tcn_channels,
            kernel_sizes=tcn_kernel_sizes,
            dilations=tcn_dilations,
            dropout=dropout,
        )

        encoder_hidden_size = int(
            getattr(self.encoder, 'output_size', 0)
            or getattr(self.encoder, 'hidden_size', 0)
        )
        if encoder_hidden_size <= 0:
            raise ValueError("failed to infer encoder hidden size")
        self.encoder_hidden_size = encoder_hidden_size

        feature_size = self.encoder_hidden_size
        if self.use_decoder:
            decoder_size = int(getattr(self.decoder, "mamba_d_model", feature_size))
            self.branch_fusion = nn.Linear(
                self.encoder_hidden_size + self.tcn_channels,
                decoder_size,
            )
            feature_size = decoder_size
        else:
            self.branch_fusion = None
        self.interval_head = nn.Linear(feature_size, 3)
        self.gaussian_head = nn.Linear(feature_size, 1)

        if not self.use_decoder:
            self.encoder_regressor = nn.Linear(self.encoder_hidden_size, 1)
        else:
            self.encoder_regressor = None

    def forward(self, encoder_x):
        use_decoder = bool(getattr(self, 'use_decoder', True))

        if self.use_spatial_gat:
            encoder_x = self.spatial_gat(encoder_x)

        encoder_output, encoder_state = self.encoder(encoder_x)
        # TCN 分支提取短、中、长时间尺度特征，与 Mamba 编码器分支并行计算。
        tcn_output = self.tcn_fusion(encoder_x)

        if use_decoder:
            # 逐时间步拼接两条分支，再压缩到解码器所需的 d_model。
            decoder_input = self.branch_fusion(
                torch.cat((encoder_output, tcn_output), dim=-1)
            )
            fused_features, attention_weights = self.decoder(decoder_input, encoder_state)
        else:
            fused_features = encoder_output[:, -1, :]
            attention_weights = torch.zeros(
                encoder_output.size(0),
                encoder_output.size(1),
                device=encoder_output.device,
                dtype=encoder_output.dtype,
            )

        interval_params = self.interval_head(fused_features)
        lower = torch.sigmoid(interval_params[:, 0:1])
        width = torch.sigmoid(interval_params[:, 1:2]) * (1.0 - lower)
        upper = lower + width
        weight = torch.sigmoid(interval_params[:, 2:3])
        point = weight * upper + (1.0 - weight) * lower
        gaussian_head = getattr(self, "gaussian_head", None)
        if gaussian_head is None:
            # Older PIVEN checkpoints predate the Gaussian uncertainty head.
            # Keep them usable by deriving a finite fallback scale from width.
            sigma = (upper - lower).detach().clamp_min(1e-4) / 3.92
        else:
            sigma = torch.nn.functional.softplus(gaussian_head(fused_features)) + 1e-4
        return point, {
            "lower": lower,
            "upper": upper,
            "weight": weight,
            "sigma": sigma,
            "features": fused_features,
        }
