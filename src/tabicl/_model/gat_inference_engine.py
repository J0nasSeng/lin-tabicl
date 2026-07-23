"""Inference helpers for frozen graph-backend TabICL checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .graph import SparseGraphSet, build_class_conditioned_graphs
from .inference_config import InferenceConfig
from .tabicl import TabICL


class GATInferenceEngine(nn.Module):
	"""Run a frozen graph-backend :class:`TabICL` model.

	The engine deliberately wraps the existing model instead of changing its
	implementation.  ``mode='ensemble'`` performs one normal model pass.  In
	``mode='reasoning'``, the graph-column representation is passed through a
	selected suffix of the existing GAT stack repeatedly.

	Parameters
	----------
	checkpoint_path : str or Path
		Checkpoint containing ``config`` and ``state_dict``.
	device : str or torch.device, optional
		Target device. Defaults to CUDA when available.
	mode : {'ensemble', 'reasoning'}
		Inference mode.
	num_iterations : int, default=1
		Number of reasoning passes. It is ignored in ensemble mode.
	entry_layer : int, optional
		One-based GAT block at which reasoning starts. The suffix from this
		block through the final block is repeated. ``None`` repeats the full
		stack.
	"""

	def __init__(
		self,
		checkpoint_path: str | Path,
		device: str | torch.device | None = None,
		mode: str = "ensemble",
		num_iterations: int = 1,
		entry_layer: int | str | None = None,
	) -> None:
		super().__init__()
		if mode not in {"ensemble", "reasoning"}:
			raise ValueError("mode must be 'ensemble' or 'reasoning'")
		if num_iterations <= 0:
			raise ValueError("num_iterations must be positive")

		self.mode = mode
		self.num_iterations = int(num_iterations)
		if entry_layer not in (None, "last") and not isinstance(entry_layer, int):
			raise ValueError("entry_layer must be an integer, 'last', or None")
		self.entry_layer = entry_layer
		self.device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

		checkpoint_path = Path(checkpoint_path)
		checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
		if "config" not in checkpoint or "state_dict" not in checkpoint:
			raise ValueError("The checkpoint must contain 'config' and 'state_dict'.")

		config = dict(checkpoint["config"])
		# Training checkpoints include this discriminator, while ``TabICL``
		# accepts the architecture parameters directly.
		config.pop("model_type", None)
		# Older graph checkpoints may predate persisting this architecture flag;
		# infer it from the learned residual parameters so their weights load with
		# the same module structure.
		if "learnable_residual" not in config:
			config["learnable_residual"] = any(
				key.endswith("attn.alpha") for key in checkpoint["state_dict"]
			)
		if config.get("icl_backend", "graph") != "graph":
			raise ValueError("GATInferenceEngine requires a checkpoint with icl_backend='graph'.")

		self.model_config_ = config
		self.model = TabICL(**config)
		self.model.load_state_dict(checkpoint["state_dict"])
		self.model.to(self.device_).eval()
		for parameter in self.model.parameters():
			parameter.requires_grad_(False)

		self.model_path_ = checkpoint_path
		self.inference_config_ = InferenceConfig()

		gat = self.model.icl_predictor.gat_icl
		self.num_layers = len(gat.graph_blocks)
		if self.entry_layer == "last":
			self.entry_layer = self.num_layers
		if self.entry_layer is not None and not 1 <= self.entry_layer <= self.num_layers:
			raise ValueError(f"entry_layer must be between 1 and {self.num_layers}")

	@property
	def max_classes(self) -> int:
		"""Expose the wrapped model's class limit for sklearn adapters."""
		return self.model.max_classes

	def _make_graph_set(self, y_train: Tensor, total_nodes: int) -> SparseGraphSet:
		predictor = self.model.icl_predictor
		return build_class_conditioned_graphs(
			y_train=y_train.long(),
			total_nodes=total_nodes,
			num_graphs=predictor.graph_num_graphs,
			min_train_neighbors=predictor.graph_min_train_neighbors,
			max_train_neighbors=predictor.graph_max_train_neighbors,
			same_label_ratio=predictor.graph_same_label_ratio,
			cross_label_ratio=predictor.graph_cross_label_ratio,
			test_k_per_class=predictor.graph_test_k_per_class,
			seed=predictor.graph_seed,
			share_graph_across_batch=predictor.graph_share_across_batch,
			share_graph_require_identical_labels=predictor.graph_share_require_identical_labels,
		)

	def _graph_input(
		self,
		X: Tensor,
		y_train: Tensor,
		inference_config: InferenceConfig,
		feature_shuffles: Optional[Sequence[Sequence[int]]] = None,
	) -> Tensor:
		col_embeddings = self.model.col_embedder(
			X,
			y_train=y_train,
			embed_with_test=False,
			feature_shuffles=feature_shuffles,
			mgr_config=inference_config.COL_CONFIG,
		)
		pre_col_embeddings = self.model.col_embedder.project_input(X)
		return self.model.icl_predictor.prepare_graph_input(
			col_embeddings=col_embeddings,
			y_train=y_train,
			pre_col_embeddings=pre_col_embeddings,
		)

	def _run_refinement(self, graph_input: Tensor, graph_set: SparseGraphSet) -> Tensor:
		predictor = self.model.icl_predictor
		gat = predictor.gat_icl
		start = 0 if self.entry_layer is None else self.entry_layer - 1
		layers_per_graph = gat.layers_per_graph
		graph_edges = [[edge.to(graph_input.device, dtype=torch.long) for edge in graph.edge_index] for graph in graph_set.graphs]

		out = graph_input
		# Establish the representation at the requested entry point once.
		for block_idx in range(start):
			out = gat.graph_blocks[block_idx](out, graph_edges[block_idx // layers_per_graph])
			B, T, C, D = out.shape
			flat = out.reshape(B * T, C, D)
			attn_in = gat.col_attn_ln[block_idx](flat)
			attn_out, _ = gat.col_attn[block_idx](attn_in, attn_in, attn_in, need_weights=False)
			out = (flat + attn_out).reshape(B, T, C, D)

		for _ in range(self.num_iterations):
			for block_idx in range(start, len(gat.graph_blocks)):
				out = gat.graph_blocks[block_idx](out, graph_edges[block_idx // layers_per_graph])
				B, T, C, D = out.shape
				flat = out.reshape(B * T, C, D)
				attn_in = gat.col_attn_ln[block_idx](flat)
				attn_out, _ = gat.col_attn[block_idx](attn_in, attn_in, attn_in, need_weights=False)
				out = (flat + attn_out).reshape(B, T, C, D)

		# ``GraphAttentionTransformer.forward`` projects the final CLS columns
		# from 4D column representations to the 3D row representations consumed
		# by the decoder. The manual suffix loop above must perform the same step.
		num_output_cls = gat.num_output_cls
		if num_output_cls is None:
			return out
		cls_out = out[:, :, -num_output_cls:, :].reshape(B, T, num_output_cls * D)
		if gat.out_proj is not None:
			cls_out = gat.out_proj(cls_out)
		return cls_out

	def _decode(self, representations: Tensor, y_train: Tensor) -> Tensor:
		predictor = self.model.icl_predictor
		src = predictor.ln(representations) if predictor.norm_first else representations
		if predictor.decoder_type == "mlp":
			return predictor.decoder(src)
		if predictor.decoder_type == "soft_kmeans":
			return predictor._soft_kmeans_decoder(src, y_train, y_train.shape[1])
		if predictor.decoder_type == "rbf":
			return predictor._rbf_decoder(src, y_train, y_train.shape[1])
		return predictor._euclidean_decoder(src, y_train, y_train.shape[1])

	@torch.inference_mode()
	def forward(
		self,
		X: Tensor,
		y_train: Tensor,
		d: Tensor | None = None,
		embed_with_test: bool = False,
		feature_shuffles: Optional[Sequence[Sequence[int]]] = None,
		return_logits: bool = True,
		softmax_temperature: float = 0.9,
		graph_set: SparseGraphSet | None = None,
		inference_config: InferenceConfig | None = None,
		return_repr: bool = False,
	) -> Tensor | tuple[Tensor, Tensor]:
		"""Return predictions for a batch of already assembled tables."""
		X = X.to(self.device_)
		y_train = y_train.to(self.device_)
		inference_config = inference_config or self.inference_config_
		if graph_set is None:
			graph_set = self._make_graph_set(y_train, X.shape[1])

		if self.mode == "ensemble" or self.num_iterations == 1 and self.entry_layer is None:
			if return_repr:
				graph_input = self._graph_input(X, y_train, inference_config, feature_shuffles)
				representation = self._run_refinement(graph_input, graph_set)
				out = self._decode(representation, y_train)[:, y_train.shape[1] :]
				if self.model.max_classes > 0:
					num_classes = int(torch.unique(y_train[0]).numel())
					out = out[..., :num_classes]
				if not return_logits:
					if self.model.icl_predictor.decoder_type in {"soft_kmeans", "rbf", "euclidean"}:
						out = out.exp()
					else:
						out = torch.softmax(out / softmax_temperature, dim=-1)
				return out, representation
			out = self.model._inference_forward(
				X=X,
				y_train=y_train,
				feature_shuffles=feature_shuffles,
				embed_with_test=embed_with_test,
				return_logits=return_logits,
				softmax_temperature=softmax_temperature,
				inference_config=inference_config,
				graph_set=graph_set,
			)
			return (out, None) if return_repr else out

		graph_input = self._graph_input(X, y_train, inference_config, feature_shuffles)
		refined = self._run_refinement(graph_input, graph_set)
		out = self._decode(refined, y_train)[:, y_train.shape[1] :]
		if self.model.max_classes > 0:
			# Match ICLearning's inference path: ensemble label shuffles operate
			# on the classes present in this dataset, not the checkpoint maximum.
			num_classes = int(torch.unique(y_train[0]).numel())
			out = out[..., :num_classes]
		if not return_logits:
			if self.model.icl_predictor.decoder_type in {"soft_kmeans", "rbf", "euclidean"}:
				out = out.exp()
			else:
				out = torch.softmax(out / softmax_temperature, dim=-1)
		return (out, refined) if return_repr else out
