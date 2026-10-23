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
	"""Single-head sensor GAT with configurable depth (1/2/3 layers)."""

	def __init__(self,
				 num_sensors=14,
				 in_features=1,
				 hidden_features=8,
				 out_features=16,
				 num_layers=2,
				 embed_dim=16,
				 topk=5,
				 graph_mode="dynamic_knn",
				 dropout=0.0,
				 alpha=0.1):
		super(SensorSpatialGAT, self).__init__()
		valid_graph_modes = ("dynamic_knn", "path")
		if graph_mode not in valid_graph_modes:
			raise ValueError(
				f"graph_mode must be one of {valid_graph_modes}, but got {graph_mode}."
			)
		if num_layers not in (1, 2, 3):
			raise ValueError("num_layers must be one of [1, 2, 3].")
		self.num_sensors = num_sensors
		self.out_features = out_features
		self.num_layers = num_layers
		self.embed_dim = embed_dim
		self.graph_mode = graph_mode
		if topk is None:
			topk = num_sensors
		self.topk = max(1, min(int(topk), num_sensors))
		self.sensor_embedding = nn.Embedding(num_sensors, embed_dim)
		nn.init.kaiming_uniform_(self.sensor_embedding.weight, a=math.sqrt(5))
		self.latest_cosine_similarity = None
		self.latest_adjacency = None
		self._static_cosine_similarity = None
		self._static_adjacency = None
		layer_dims = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]
		self.gat_layers = nn.ModuleList([
			GraphAttentionLayer(
				in_features=layer_dims[idx],
				out_features=layer_dims[idx + 1],
				dropout=dropout,
				alpha=alpha,
			)
			for idx in range(num_layers)
		])
		# Keep legacy attribute names to reduce friction with old utilities/checkpoints.
		self.gat1 = self.gat_layers[0]
		if num_layers >= 2:
			self.gat2 = self.gat_layers[1]
		if num_layers >= 3:
			self.gat3 = self.gat_layers[2]
		self.activation = nn.ELU()
		# Preserve node-specific information before compression.
		self.gat_fusion = nn.Linear(num_sensors * out_features, out_features)
		# Residual path from raw sensor values to avoid over-smoothing collapse.
		self.raw_projection = nn.Linear(num_sensors, out_features)
		self.gate_projection = nn.Linear(out_features * 2, out_features)
		self.output_norm = nn.LayerNorm(out_features)

	def _compute_cosine_similarity(self, device):
		node_indices = torch.arange(self.num_sensors, device=device)
		node_embeddings = self.sensor_embedding(node_indices)
		normed_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
		return torch.matmul(normed_embeddings, normed_embeddings.transpose(0, 1))

	def _build_knn_adjacency(self, cosine_similarity):
		topk_indices = torch.topk(cosine_similarity, self.topk, dim=-1)[1]
		adjacency = torch.zeros(
			self.num_sensors,
			self.num_sensors,
			device=cosine_similarity.device,
			dtype=cosine_similarity.dtype,
		)
		adjacency.scatter_(1, topk_indices, 1.0)
		adjacency.fill_diagonal_(1.0)
		return adjacency

	def _build_path_adjacency(self, device, dtype):
		adjacency = torch.eye(self.num_sensors, device=device, dtype=dtype)
		if self.num_sensors > 1:
			idx = torch.arange(self.num_sensors - 1, device=device)
			adjacency[idx, idx + 1] = 1.0
			adjacency[idx + 1, idx] = 1.0
		return adjacency

	def _build_adjacency(self, device):
		"""Build graph adjacency according to graph_mode."""
		if self.graph_mode == "dynamic_knn":
			cosine_similarity = self._compute_cosine_similarity(device)
			adjacency = self._build_knn_adjacency(cosine_similarity)
		elif self.graph_mode == "path":
			if self._static_adjacency is None or self._static_adjacency.device != device:
				self._static_adjacency = self._build_path_adjacency(
					device=device,
					dtype=self.sensor_embedding.weight.dtype,
				)
				self._static_cosine_similarity = None
			adjacency = self._static_adjacency
			cosine_similarity = self._static_cosine_similarity
		else:
			raise ValueError(f"Unsupported graph_mode: {self.graph_mode}")

		if cosine_similarity is None:
			self.latest_cosine_similarity = None
		else:
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
		adjacency = self._build_adjacency(x.device)
		adjacency = adjacency.unsqueeze(0).expand(batch_size * seq_len, -1, -1)

		if hasattr(self, "gat_layers"):
			h = node_features
			for gat_layer in self.gat_layers:
				h = gat_layer(h, adjacency)
				h = self.activation(h)
		else:
			# Legacy fallback for old checkpoints serialized before gat_layers existed.
			h = self.gat1(node_features, adjacency)
			h = self.activation(h)
			if hasattr(self, "gat2"):
				h = self.gat2(h, adjacency)
				h = self.activation(h)
			if hasattr(self, "gat3"):
				h = self.gat3(h, adjacency)
				h = self.activation(h)

		# 节点保留融合：先将所有节点嵌入向量展平，然后进行投影
		gat_fused = h.reshape(batch_size * seq_len, num_sensors * self.out_features)
		gat_fused = self.gat_fusion(gat_fused).reshape(batch_size, seq_len, self.out_features)

		# 残差门控融合技术保留原始信息+空间信息。
		raw_fused = self.raw_projection(x)
		gate = torch.sigmoid(self.gate_projection(torch.cat((gat_fused, raw_fused), dim=-1)))
		fused = gate * gat_fused + (1.0 - gate) * raw_fused
		return self.output_norm(fused)
