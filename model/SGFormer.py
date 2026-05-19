import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul


def _expand_edge_index_for_batch(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    """Repeat a single-graph edge_index into a block-diagonal batched edge_index.

    Args:
        edge_index: LongTensor [2, E] with node ids in [0, num_nodes).
        batch_size: number of independent graphs.
        num_nodes: nodes per graph.

    Returns:
        LongTensor [2, batch_size * E] with node ids in [0, batch_size * num_nodes).
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()
    if batch_size <= 1:
        return edge_index

    E = edge_index.size(1)
    device = edge_index.device
    offsets = (torch.arange(batch_size, device=device, dtype=torch.long)
               .repeat_interleave(E) * int(num_nodes))
    return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)


def full_attention_conv(qs, ks, vs, output_attn=False):
    """
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    """
    # normalize input
    # Support both unbatched ([N, H, M]) and batched ([B, N, H, M]) forms.
    if qs.dim() == 3:
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]
        batched = False
    elif qs.dim() == 4:
        # Normalize per batch item to avoid mixing graphs.
        qs = qs / torch.norm(qs, p=2, dim=(1, 2, 3), keepdim=True)  # [B, N, H, M]
        ks = ks / torch.norm(ks, p=2, dim=(1, 2, 3), keepdim=True)  # [B, L, H, M]
        N = qs.shape[1]
        batched = True
    else:
        raise ValueError(f"qs must be 3D or 4D, got shape {tuple(qs.shape)}")

    # numerator
    if not batched:
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs
    else:
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [B, N, H, D]
        attention_num += N * vs

    # denominator
    if not batched:
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]
    else:
        all_ones = torch.ones([ks.shape[1]], device=ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [B, N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape)
    )  # [N, H, 1] or [B, N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D] or [B, N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        if not batched:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, L]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(
                dim=-1, keepdim=True
            )  # [N,1]
            attention = attention / normalizer
        else:
            attention = torch.einsum("bnhm,blhm->bnlh", qs, ks).mean(dim=-1)  # [B, N, L]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(
                dim=-1, keepdim=True
            )  # [B, N, 1]
            attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1.0 / d[col]).sqrt()
        d_norm_out = (1.0 / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_init=False,
        use_act=True,
    ):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, output_attn=False):
        # feature transformation
        if query_input.dim() == 2:
            query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
            key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
            if self.use_weight:
                value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
            else:
                value = source_input.reshape(-1, 1, self.out_channels)
        elif query_input.dim() == 3:
            # [B, N, D] -> [B, N, H, C]
            query = self.Wq(query_input).reshape(query_input.size(0), query_input.size(1), self.num_heads, self.out_channels)
            key = self.Wk(source_input).reshape(source_input.size(0), source_input.size(1), self.num_heads, self.out_channels)
            if self.use_weight:
                value = self.Wv(source_input).reshape(source_input.size(0), source_input.size(1), self.num_heads, self.out_channels)
            else:
                value = source_input.reshape(source_input.size(0), source_input.size(1), 1, self.out_channels)
        else:
            raise ValueError(f"query_input must be 2D or 3D, got shape {tuple(query_input.shape)}")

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn
            )  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value)  # [N, H, D]

        # attention_output: [N, H, D] or [B, N, H, D]
        if attention_output.dim() == 3:
            final_output = attention_output.mean(dim=1)
        else:
            final_output = attention_output.mean(dim=2)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        alpha=0.5,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_act=True,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(
                    hidden_channels,
                    hidden_channels,
                    num_heads=num_heads,
                    use_weight=use_weight,
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        # self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index=None):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N] or [layer num, B, N, N]


class SGFormer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        trans_num_layers=1,
        trans_num_heads=1,
        trans_dropout=0.5,
        gnn_num_layers=1,
        gnn_dropout=0.5,
        gnn_use_weight=True,
        gnn_use_init=False,
        gnn_use_bn=True,
        gnn_use_residual=True,
        gnn_use_act=True,
        alpha=0.5,
        trans_use_bn=True,
        trans_use_residual=True,
        trans_use_weight=True,
        trans_use_act=True,
        use_graph=True,
        graph_weight=0.8,
        aggregate="add",
    ):
        super().__init__()
        self.trans_conv = TransConv(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            alpha,
            trans_dropout,
            trans_use_bn,
            trans_use_residual,
            trans_use_weight,
            trans_use_act,
        )
        self.graph_conv = GraphConv(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == "add":
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == "cat":
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f"Invalid aggregate type:{aggregate}")

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = (
            list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        )
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        x1 = self.trans_conv(x)
        if self.use_graph:
            if edge_index is None:
                raise ValueError("edge_index cannot be None when use_graph=True")

            if x.dim() == 3:
                # x: [B, N, D] -> flatten to [B*N, D] for GraphConv, with block-diagonal edges.
                B, N, D = x.shape
                # If user provided a single-graph edge_index ([0, N)), expand it.
                if int(edge_index.max().item()) < int(N):
                    edge_index = _expand_edge_index_for_batch(edge_index, batch_size=B, num_nodes=N)
                x2 = self.graph_conv(x.reshape(B * N, D), edge_index)
                x2 = x2.view(B, N, -1)
            else:
                x2 = self.graph_conv(x, edge_index)

            if self.aggregate == "add":
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=-1 if x1.dim() == 3 else 1)
        else:
            x = x1
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()