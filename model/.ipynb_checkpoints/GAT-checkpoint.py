import math
import torch
import torch.nn.functional as F
from torch import nn


class GraphAttentionLayer(nn.Module):
	"""Single-head graph attention layer for dense adjacency matrices."""

	def __init__(self, in_features, out_features, dropout=0.0, alpha=0.1):
		super(GraphAttentionLayer, self).__init__()
		self.linear = nn.Linear(in_features, out_features, bias=False)
		self.attn_src = nn.Linear(out_features, 1, bias=False)
		self.attn_dst = nn.Linear(out_features, 1, bias=False)
		self.leaky_relu = nn.LeakyReLU(alpha)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, adjacency):
		# x: (batch_like, num_nodes, in_features)
		h = self.linear(x)
		src_score = self.attn_src(h)
		dst_score = self.attn_dst(h)
		e = self.leaky_relu(src_score + dst_score.transpose(1, 2))

		masked_e = e.masked_fill(adjacency <= 0, float("-inf"))
		attention = torch.softmax(masked_e, dim=-1)
		attention = self.dropout(attention)
		return torch.bmm(attention, h)


class SensorSpatialGAT(nn.Module):
	"""Two-layer single-head GAT for 14-sensor fully connected graphs."""

	def __init__(self,
				 num_sensors=14,
				 in_features=1,
				 hidden_features=8,
				 out_features=16,
				 embed_dim=16,
				 topk=5,
				 dropout=0.0,
				 alpha=0.1):
		super(SensorSpatialGAT, self).__init__()
		self.num_sensors = num_sensors
		self.out_features = out_features
		self.embed_dim = embed_dim
		if topk is None:
			topk = num_sensors
		self.topk = max(1, min(int(topk), num_sensors))
		self.sensor_embedding = nn.Embedding(num_sensors, embed_dim)
		nn.init.kaiming_uniform_(self.sensor_embedding.weight, a=math.sqrt(5))
		self.latest_cosine_similarity = None
		self.latest_adjacency = None
		self.gat1 = GraphAttentionLayer(
			in_features=in_features,
			out_features=hidden_features,
			dropout=dropout,
			alpha=alpha,
		)
		self.gat2 = GraphAttentionLayer(
			in_features=hidden_features,
			out_features=out_features,
			dropout=dropout,
			alpha=alpha,
		)
		self.activation = nn.ELU()
		# Preserve node-specific information before compression.
		self.gat_fusion = nn.Linear(num_sensors * out_features, out_features)
		# Residual path from raw sensor values to avoid over-smoothing collapse.
		self.raw_projection = nn.Linear(num_sensors, out_features)
		self.gate_projection = nn.Linear(out_features * 2, out_features)
		self.output_norm = nn.LayerNorm(out_features)

	def _build_dynamic_adjacency(self, device):
		"""Build directed adjacency by top-k cosine similarity between sensor embeddings."""
		# 生成传感器节点的索引序列 [0,1,2,...,num_sensors-1]
		node_indices = torch.arange(self.num_sensors, device=device)
		# 通过嵌入层，把每个传感器索引映射为低维向量（学习到的特征表示）
		node_embeddings = self.sensor_embedding(node_indices)
		# 对嵌入向量做L2归一化，方便计算余弦相似度
		normed_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
		# 计算所有传感器两两之间的余弦相似度矩阵
		cosine_similarity = torch.matmul(normed_embeddings, normed_embeddings.transpose(0, 1))
		# 对每个传感器，找出相似度最高的topk个其他传感器
		topk_indices = torch.topk(cosine_similarity, self.topk, dim=-1)[1]
		# 初始化全0的邻接矩阵
		adjacency = torch.zeros(
			self.num_sensors,
			self.num_sensors,
			device=device,
			dtype=cosine_similarity.dtype,
		)
		# 把top-k最相似的节点位置赋值为1、建立有向连接
		adjacency.scatter_(1, topk_indices, 1.0)
		adjacency.fill_diagonal_(1.0)

		self.latest_cosine_similarity = cosine_similarity.detach()
		self.latest_adjacency = adjacency.detach()
		return adjacency

	def forward(self, x):
		# x: (batch, seq_len, num_sensors)
		batch_size, seq_len, num_sensors = x.shape
		if num_sensors != self.num_sensors:
			raise ValueError(
				f"SensorSpatialGAT expects {self.num_sensors} sensors, but got {num_sensors}."
			)

		# Process each time step independently as a sensor graph.
		node_features = x.reshape(batch_size * seq_len, num_sensors, 1)
		adjacency = self._build_dynamic_adjacency(x.device)
		adjacency = adjacency.unsqueeze(0).expand(batch_size * seq_len, -1, -1)

		h = self.gat1(node_features, adjacency)
		h = self.activation(h)
		h = self.gat2(h, adjacency)
		h = self.activation(h)

		# 节点保留融合：先将所有节点嵌入向量展平，然后进行投影
		gat_fused = h.reshape(batch_size * seq_len, num_sensors * self.out_features)
		gat_fused = self.gat_fusion(gat_fused).reshape(batch_size, seq_len, self.out_features)

		# 残差门控融合技术保留原始信息+空间信息。
		raw_fused = self.raw_projection(x)
		gate = torch.sigmoid(self.gate_projection(torch.cat((gat_fused, raw_fused), dim=-1)))
		fused = gate * gat_fused + (1.0 - gate) * raw_fused
		return self.output_norm(fused)
