"""Inference helpers for frozen graph-backend TabICL checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .graph import (
	CompactGraphSet,
	SparseGraphBatch,
	SparseGraphSet,
	build_class_conditioned_graphs,
	GraphPrior,
	stack_graph_sets,
)
from .inference_config import InferenceConfig
from .tabicl import TabICL
from .gat import GraphMultiheadAttention
from .gat_pyg import GraphMultiheadAttention as PyGGraphMultiheadAttention


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
		max_chunk_size: int | None = None,
		decoder_chunk_size: int | None = None,
	) -> None:
		super().__init__()
		if mode not in {"ensemble", "reasoning"}:
			raise ValueError("mode must be 'ensemble' or 'reasoning'")
		if num_iterations <= 0:
			raise ValueError("num_iterations must be positive")
		if decoder_chunk_size is not None and decoder_chunk_size <= 0:
			raise ValueError("decoder_chunk_size must be positive")

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
		if config.get("icl_backend", "graph") not in {
			"graph", "graph-pyg", "graph-2d", "graph-2d-pyg",
			"graph-1d", "graph-1d-pyg",
		}:
			raise ValueError("GATInferenceEngine requires a graph checkpoint.")

		self.model_config_ = config
		self.model = TabICL(**config)
		self.model.load_state_dict(checkpoint["state_dict"])
		if decoder_chunk_size is not None:
			self.model.decoder_chunk_size = int(decoder_chunk_size)
			self.model.icl_predictor.decoder_chunk_size = int(decoder_chunk_size)
		if max_chunk_size is not None:
			if max_chunk_size <= 0:
				raise ValueError("max_chunk_size must be > 0 when provided")
			for module in self.model.modules():
				if isinstance(module, (GraphMultiheadAttention, PyGGraphMultiheadAttention)):
					module.max_chunk_size = int(max_chunk_size)
		self.model.to(self.device_).eval()
		for parameter in self.model.parameters():
			parameter.requires_grad_(False)

		self.model_path_ = checkpoint_path
		self.inference_config_ = InferenceConfig()
		self.max_features = self.model.max_features

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
		prior = GraphPrior(
			graph_v1_prob=getattr(predictor, "graph_v1_prob", 1.0),
			graph_v2_prob=getattr(predictor, "graph_v2_prob", 0.0),
			graph_prob=getattr(predictor, "graph_prob", 0.0),
			min_train_neighbors=predictor.graph_min_train_neighbors,
			max_train_neighbors=predictor.graph_max_train_neighbors,
			cross_label_fraction=predictor.graph_cross_label_fraction,
			train_neighbors_per_test=predictor.graph_train_neighbors_per_test,
			seed=predictor.graph_seed,
			share_graph_across_batch=predictor.graph_share_across_batch,
		)
		full_labels = torch.zeros((y_train.shape[0], total_nodes), dtype=y_train.dtype, device=y_train.device)
		full_labels[:, : y_train.shape[1]] = y_train
		return prior(full_labels.long(), y_train.shape[1], num_graphs=predictor.graph_num_graphs)

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
		if self.model.icl_backend in {"graph-1d", "graph-1d-pyg"}:
			return self.model.row_interactor(
				col_embeddings,
				mgr_config=inference_config.ROW_CONFIG,
			)
		pre_col_embeddings = self.model.col_embedder.project_input(X)
		return self.model.icl_predictor.prepare_graph_input(
			col_embeddings=col_embeddings,
			y_train=y_train,
			pre_col_embeddings=pre_col_embeddings,
		)

	def _run_refinement(self, graph_input: Tensor, graph_set: SparseGraphSet | CompactGraphSet) -> Tensor:
		predictor = self.model.icl_predictor
		gat = predictor.gat_icl
		start = 0 if self.entry_layer is None else self.entry_layer - 1
		layers_per_graph = gat.layers_per_graph
		if hasattr(graph_set, "edge_offsets") and hasattr(graph_set, "edge_index"):
			# Keep compact graph payloads compact.  The compatibility ``graphs``
			# property materializes Python lists and, when offsets are on CUDA, can
			# also synchronize while a previous invalid index is still pending.
			# Convert each dataset slice to local indices for the graph blocks.
			batch_size = graph_input.shape[0]
			compact_edges = graph_set.edge_index.to(graph_input.device, dtype=torch.long)
			graph_edges = []
			for graph_idx in range(graph_set.num_graphs):
				starts = graph_set.edge_offsets[graph_idx, :-1].detach().cpu().tolist()
				ends = graph_set.edge_offsets[graph_idx, 1:].detach().cpu().tolist()
				graph_edges.append([
					compact_edges[:, start:end]
					for start, end in zip(starts, ends)
				])
			if len(graph_edges[0]) != batch_size:
				raise ValueError("Compact graph batch size does not match graph input")
		else:
			graph_edges = [
				[edge.to(graph_input.device, dtype=torch.long) for edge in graph.edge_index]
				for graph in graph_set.graphs
			]

		# The sklearn/AutoGluon adapter assembles inputs as float32, while some
		# checkpoints are loaded with half-precision GAT weights. Match the
		# activations to the graph stack before invoking LayerNorm or attention.
		gat_dtype = next(gat.parameters()).dtype
		out = graph_input.to(dtype=gat_dtype)
		if self.model.icl_backend in {"graph-1d", "graph-1d-pyg"}:
			for block_idx in range(start):
				out = gat.graph_blocks[block_idx](
					out.unsqueeze(2), graph_edges[block_idx // layers_per_graph]
				).squeeze(2)
			for _ in range(self.num_iterations):
				for block_idx in range(start, len(gat.graph_blocks)):
					out = gat.graph_blocks[block_idx](
						out.unsqueeze(2), graph_edges[block_idx // layers_per_graph]
					).squeeze(2)
			return out

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

	def _validate_graph_set(self, graph_set: SparseGraphSet | CompactGraphSet, total_nodes: int, batch_size: int) -> None:
		"""Validate externally supplied graph metadata before launching CUDA kernels."""
		gat = self.model.icl_predictor.gat_icl
		if graph_set.num_graphs != gat.num_graphs:
			raise ValueError(
				f"Expected {gat.num_graphs} graphs for the GAT stack, got {graph_set.num_graphs}"
			)
		if graph_set.num_nodes != total_nodes:
			raise ValueError(f"Expected graph num_nodes={total_nodes}, got {graph_set.num_nodes}")
		if isinstance(graph_set, CompactGraphSet):
			if graph_set.num_datasets != batch_size:
				raise ValueError(
					f"Expected {batch_size} graph datasets, got {graph_set.num_datasets}"
				)
			edge_index = graph_set.edge_index
		else:
			if any(len(graph.edge_index) != batch_size for graph in graph_set.graphs):
				raise ValueError("Each graph must contain one edge index per input batch item")
			edge_index = torch.cat(
				[edge for graph in graph_set.graphs for edge in graph.edge_index], dim=1
			) if graph_set.graphs else torch.empty((2, 0), dtype=torch.long)
		if edge_index.ndim != 2 or edge_index.shape[0] != 2:
			raise ValueError("Graph edge indices must have shape (2, num_edges)")
		if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= total_nodes):
			raise ValueError("Graph edge indices must be within the input node range")

	def _compact_graph_set(
		self, graph_set: SparseGraphSet | CompactGraphSet, batch_size: int
	) -> CompactGraphSet:
		if isinstance(graph_set, CompactGraphSet):
			return graph_set
		if not graph_set.graphs:
			raise ValueError("Graph set must contain at least one graph")
		graph_batch_sizes = {len(graph.edge_index) for graph in graph_set.graphs}
		if graph_batch_sizes == {1}:
			return stack_graph_sets([graph_set])
		if graph_batch_sizes != {batch_size}:
			raise ValueError("Sparse graph batches must all match the input batch size")
		per_dataset = [
			SparseGraphSet(
				graphs=[
					SparseGraphBatch(edge_index=[graph.edge_index[batch_idx]], num_nodes=graph.num_nodes)
					for graph in graph_set.graphs
				]
			)
			for batch_idx in range(batch_size)
		]
		return stack_graph_sets(per_dataset)

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
		graph_set: SparseGraphSet | CompactGraphSet | None = None,
		inference_config: InferenceConfig | None = None,
		return_repr: bool = False,
	) -> Tensor | tuple[Tensor, Tensor]:
		"""Return predictions for a batch of already assembled tables."""
		X = X.to(self.device_)
		y_train = y_train.to(self.device_)
		if self.model.max_classes > 0 and int(torch.unique(y_train[0]).numel()) > self.model.max_classes:
			if self.mode != "ensemble":
				raise ValueError("Hierarchical graph inference requires mode='ensemble'")
		inference_config = inference_config or self.inference_config_
		if graph_set is None:
			graph_set = self._make_graph_set(y_train, X.shape[1])
		else:
			graph_set = self._compact_graph_set(graph_set, X.shape[0])
			self._validate_graph_set(graph_set, X.shape[1], X.shape[0])

		is_graph_1d = self.model.icl_backend in {"graph-1d", "graph-1d-pyg"}
		if (
			(self.mode == "ensemble" and not is_graph_1d)
			or (self.num_iterations == 1 and self.entry_layer is None and not is_graph_1d)
		):
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
