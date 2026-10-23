import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Tcn_DW_1d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, n_groups ,padding, dropout=0.2):
        super(Tcn_DW_1d, self).__init__()
        # self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                                   stride=stride, padding=padding, dilation=dilation))

        # 创建深度卷积层，输出通道数为 C*K
        self.depthwise_conv = weight_norm(nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=n_groups  # 关键：分组数等于输入通道数
            )
        )

        self.chomp = Chomp1d(padding)
        self.leakyrelu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.depthwise_conv, self.chomp, self.leakyrelu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.depthwise_conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, out_channel, seq_len)
        :return:size of (Batch, out_channel, seq_len)
        """
        out = self.net(x)
        return out

class TCN_DW_G(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):  # k>=3

        super(TCN_DW_G, self).__init__()

        num_channels = [in_channels, in_channels*3, in_channels*5, out_channels]  # 设定不同层的通道数
        num_levels = len(num_channels) - 1  # 计算层数

        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i  # Expansion coefficient: 1，2，4，8……
            layers += [Tcn_DW_1d(num_channels[i], num_channels[i + 1], kernel_size, stride=1, dilation=dilation_size,
                                      padding=(kernel_size - 1) * dilation_size, n_groups= in_channels)]

        self.network = nn.Sequential(*layers)
    def forward(self, x):
        B, M, L = x.shape
        x = self.network(x)
        out = x.reshape(B, M, -1, L)
        return out

class TCN_DW_L(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):  # k>=3

        super(TCN_DW_L, self).__init__()

        num_channels = [in_channels, in_channels*3, in_channels*5, out_channels]  # 设定不同层的通道数
        num_levels = len(num_channels) - 1  # 计算层数

        layers = []
        for i in range(num_levels):
            dilation_size = 1  # Expansion coefficient: 1，2，4，8……
            layers += [Tcn_DW_1d(num_channels[i], num_channels[i + 1], kernel_size, stride=1, dilation=dilation_size,
                                      padding=(kernel_size - 1) * dilation_size, n_groups= in_channels)]

        self.network = nn.Sequential(*layers)
    def forward(self, x):
        B, M, L = x.shape
        x = self.network(x)
        out = x.reshape(B, M, -1, L)
        return out



class MAD_TCN(nn.Module):
    def __init__(self, dmodel, nvars, n_window, ffn_ratio=2, drop=0.1, alpha_value=0.5):
        super(MAD_TCN, self).__init__()

        # 2) 量化成 int（你可以选 round / floor / ceil）
        self.dff = max(1, int(round(ffn_ratio * dmodel)))

        if self.dff * nvars % dmodel != 0:
            raise ValueError(
                f"Group_Conv_Error:ffn_ratio and dmodel must fit,self.dff* nvars % dmodel must equal  0, got {ffn_ratio}")

        # self.dw = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
        #                                  kernel_size=3, stride=1, padding=1, groups=nvars * dmodel)
        self.dw1 = TCN_DW_L(in_channels=nvars, out_channels=nvars * dmodel // 2, kernel_size=3)
        self.norm1 = nn.BatchNorm1d(dmodel)
        self.dw2 = TCN_DW_G(in_channels=nvars, out_channels=nvars * dmodel // 2, kernel_size=3)
        self.norm2 = nn.BatchNorm1d(dmodel)
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * self.dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * self.dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * self.dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * self.dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        # self.out = nn.Linear(dmodel*dff, 1)
        # self.out = nn.Sequential(nn.Linear(dmodel*dff, 1), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(dmodel * n_window, 1), nn.Sigmoid())
        # self.alpha = nn.Parameter(torch.zeros(nvars))  # 初始化为0，sigmoid后为0.5

        # 计算 logit(init_alpha) 作为初值
        if not 0.0 < alpha_value < 1.0:
            raise ValueError("alpha_value must be between 0 and 1.")
        alpha_init = torch.logit(torch.tensor(alpha_value))
        self.alpha = nn.Parameter(alpha_init * torch.ones(nvars))

    def forward_features(self, x):
        input = x
        B, M, N = x.shape
        # x = x.reshape(B,M*D,N)
        x1 = self.dw1(x)
        x2 = self.dw2(x)
        # B, M, D, N = x.shape
        # x = x.reshape(B,M,D,N)
        #
        B, M, D, N = x1.shape
        # 沿扩展维度拼接（dim=2）
        combined = torch.cat([x1, x2], dim=2)  # (B, C, 2M, L)
        B, M, D, N = combined.shape

        x = combined.reshape(B * M, D, N)

        x = self.norm(x)
        x = x.reshape(B, M, D, N)

        x1 = x.reshape(B, M * D, N)
        x1 = self.ffn1drop1(self.ffn1pw1(x1))
        x1 = self.ffn1act(x1)
        x1 = self.ffn1drop2(self.ffn1pw2(x1))
        x1 = x1.reshape(B, M, D, N)

        # x1 = x

        x2 = x1.permute(0, 2, 1, 3)
        x2 = x2.reshape(B, D * M, N)
        x2 = self.ffn2drop1(self.ffn2pw1(x2))
        x2 = self.ffn2act(x2)
        x2 = self.ffn2drop2(self.ffn2pw2(x2))
        x2 = x2.reshape(B, D, M, N)
        x2 = x2.permute(0, 2, 1, 3)

        alpha = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        x3 = alpha * x1 + (1 - alpha) * x2
        return x3

    def forward(self, x):
        x3 = self.forward_features(x)
        B, M, D, N = x3.shape

        x3 = x3.reshape(B, M, D * N)
        out = self.out(x3)

        # x2 = x1.reshape(B, M, D*N)
        # out = self.out(x2)

        return out


class MADTCNEncoder(nn.Module):
    """MAD-TCN encoder adapter for the sequence-to-sequence Mamba decoder."""

    def __init__(self, input_size, dmodel, n_window, mamba_num_layers,
                 dropout=0.1, alpha_value=0.5):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(dmodel)
        self.hidden_size = self.output_size
        self.mamba_num_layers = int(mamba_num_layers)
        self.n_window = int(n_window)
        self.mad_tcn = MAD_TCN(
            dmodel=self.output_size,
            nvars=self.input_size,
            n_window=self.n_window,
            drop=dropout,
            alpha_value=alpha_value,
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("MADTCNEncoder expects input shaped (B, T, features).")
        if x.size(1) != self.n_window or x.size(2) != self.input_size:
            raise ValueError(
                f"MADTCNEncoder expects (T, features)=({self.n_window}, "
                f"{self.input_size}), got ({x.size(1)}, {x.size(2)})."
            )

        # MAD-TCN uses (batch, variables, window); the decoder consumes a sequence.
        features = self.mad_tcn.forward_features(x.transpose(1, 2))
        output = features.mean(dim=1).transpose(1, 2).contiguous()
        hidden_state = output[:, -1, :].unsqueeze(0).repeat(
            self.mamba_num_layers, 1, 1
        )
        cell_state = torch.zeros_like(hidden_state)
        return output, (hidden_state, cell_state)