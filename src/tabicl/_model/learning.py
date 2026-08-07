from __future__ import annotations

from typing import Literal, Optional, Union
from collections import OrderedDict
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .layers import ClassNode, OneHotAndLinear
from .encoders import Encoder
from .gat import GraphAttentionTransformer
from .graph import CompactGraphSet, SparseGraphSet
from .kv_cache import KVCache
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class ICLearning(nn.Module):
    """Dataset-wise in-context learning.

    Parameters
    ----------
    out_dim : int
        Output dimension of the model.

    max_classes : int
        Determines the task type and output behavior:
        - If max_classes=0: The model performs regression using quantile prediction.
        - If max_classes>0: The model performs classification. This value specifies
          the number of classes the model supports natively. If the number of classes
          in the dataset exceeds this value, hierarchical classification is used.

    d_model : int
        Model dimension.

    num_blocks : int
        Number of blocks used in the ICL encoder.

    nhead : int
        Number of attention heads of the ICL encoder.

    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder.

    dropout : float, default=0.0
        Dropout probability.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    ssmax : bool or str, default=False
        Type of scalable softmax to use in the ICL encoder.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:
            - "none": No scaling applied
            - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where s is learnable per-head parameter
            - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length
            - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP
            - "qassmax-mlp": Query-aware scaling: :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`
            - "qassmax-mlp-elementwise": Elementwise query-aware scaling

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        max_classes: int,
        out_dim: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
        icl_backend: Literal["encoder", "graph"] = "graph",
        graph_min_train_neighbors: int = 8,
        graph_max_train_neighbors: int = 15,
        graph_cross_label_fraction: float = 0.1,
        graph_train_neighbors_per_test: int = 8,
        decoder_type: Literal["mlp", "soft_kmeans", "rbf", "euclidean"] = "mlp",
        soft_kmeans_temperature: float = 1.0,
        graph_seed: Optional[int] = None,
        graph_share_across_batch: bool = False,
        graph_num_cls: int = 4,
        graph_num_graphs: int = 1,
        tab_graphs: str = "v1",
        mode_prob: float = 1.0,
        learnable_residual: bool = False,
    ):
        super().__init__()

        self.max_classes = max_classes
        self.norm_first = norm_first
        self.icl_backend = icl_backend
        self.graph_min_train_neighbors = graph_min_train_neighbors
        self.graph_max_train_neighbors = graph_max_train_neighbors
        self.graph_cross_label_fraction = graph_cross_label_fraction
        self.graph_train_neighbors_per_test = graph_train_neighbors_per_test
        self.decoder_type = decoder_type
        self.soft_kmeans_temperature = float(soft_kmeans_temperature)
        self.graph_seed = graph_seed
        self.graph_share_across_batch = graph_share_across_batch
        self.graph_num_cls = graph_num_cls
        self.graph_num_graphs = int(graph_num_graphs)
        self.tab_graphs = tab_graphs
        self.mode_prob = float(mode_prob)
        self.learnable_residual = bool(learnable_residual)

        if self.icl_backend not in ("encoder", "graph"):
            raise ValueError(f"Unknown icl_backend={self.icl_backend}. Expected 'encoder' or 'graph'.")

        if self.decoder_type not in ("mlp", "soft_kmeans", "rbf", "euclidean"):
            raise ValueError(
                f"Unknown decoder_type={self.decoder_type}. Expected 'mlp', 'soft_kmeans', 'rbf', or 'euclidean'."
            )

        if self.soft_kmeans_temperature <= 0:
            raise ValueError("soft_kmeans_temperature must be > 0")

        if self.graph_num_graphs <= 0:
            raise ValueError("graph_num_graphs must be positive")

        if self.icl_backend == "graph" and max_classes <= 0:
            raise ValueError("Graph ICL backend is currently supported for classification only (max_classes > 0).")

        if self.icl_backend == "encoder":
            self.tf_icl = Encoder(
                num_blocks=num_blocks,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                bias_free_ln=bias_free_ln,
                ssmax=ssmax,
                recompute=recompute,
            )
        else:
            if d_model % self.graph_num_cls != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by graph_num_cls ({self.graph_num_cls}) for graph path"
                )
            self.graph_col_dim = d_model // self.graph_num_cls
            self.graph_cls_tokens = nn.Parameter(torch.empty(self.graph_num_cls, self.graph_col_dim))
            nn.init.trunc_normal_(self.graph_cls_tokens, std=0.02)
            self.gat_icl = GraphAttentionTransformer(
                num_blocks=num_blocks,
                d_model=self.graph_col_dim,
                nhead=nhead,
                dim_feedforward=max(self.graph_col_dim * 2, dim_feedforward // max(self.graph_num_cls, 1)),
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                bias_free_ln=bias_free_ln,
                recompute=recompute,
                num_output_cls=self.graph_num_cls,
                out_dim=d_model,
                learnable_residual=self.learnable_residual,
                num_graphs=self.graph_num_graphs,
            )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model, bias=not bias_free_ln)

        if max_classes > 0:  # Classification
            self.y_encoder = OneHotAndLinear(max_classes, d_model)
        else:  # Regression
            self.y_encoder = nn.Linear(1, d_model)

        if self.decoder_type == "mlp":
            self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, out_dim))
        else:
            self.decoder = None

        enc_name = "tf_icl" if self.icl_backend == "encoder" else "gat_icl"
        self.inference_mgr = InferenceManager(enc_name=enc_name, out_dim=out_dim)

    def _aggregate_kernel_assignments(
        self,
        log_kernel: Tensor,
        y_train: Tensor,
    ) -> Tensor:
        """Aggregate train-row kernel assignments into log probabilities."""
        y_one_hot = F.one_hot(y_train.long(), num_classes=self.max_classes).bool()
        class_mask = y_one_hot.unsqueeze(1)
        log_class_mass = torch.logsumexp(
            log_kernel.unsqueeze(-1).masked_fill(~class_mask, float("-inf")),
            dim=2,
        )
        # A dataset may use fewer classes than max_classes. Give absent classes a
        # finite machine-small mass so cross_entropy with label smoothing remains
        # well-defined, then renormalize in log space.
        log_floor = math.log(torch.finfo(log_class_mass.dtype).tiny)
        return F.log_softmax(log_class_mass.clamp_min(log_floor), dim=-1)

    def _soft_kmeans_decoder(
        self, src: Tensor, y_train: Tensor, train_size: int
    ) -> Tensor:
        """Decode via train-to-all dot-product matching and class aggregation."""

        if self.max_classes <= 0:
            raise ValueError("soft_kmeans decoder is only supported for classification (max_classes > 0).")
        if train_size <= 0:
            raise ValueError("soft_kmeans decoder requires train_size > 0")

        train_repr = src[:, :train_size, :]

        # Compute similarities and the row-wise assignment distribution in float32.
        # Keeping this path in float32 avoids underflow before class aggregation.
        src_float = src.float()
        train_repr_float = train_repr.float()
        sim = torch.matmul(src_float, train_repr_float.transpose(1, 2))
        sim = sim / math.sqrt(src.shape[-1]) / self.soft_kmeans_temperature
        log_assign = torch.log_softmax(sim, dim=-1)

        return self._aggregate_kernel_assignments(log_assign, y_train)

    def _rbf_decoder(
        self, src: Tensor, y_train: Tensor, train_size: int
    ) -> Tensor:
        """Decode using an RBF kernel over train-row representations."""

        if self.max_classes <= 0:
            raise ValueError("rbf decoder is only supported for classification (max_classes > 0).")
        if train_size <= 0:
            raise ValueError("rbf decoder requires train_size > 0")

        d = torch.tensor(src.shape[-1], dtype=src.dtype, device=src.device)

        train_repr = src[:, :train_size, :]
        src_float = src.float()
        train_repr_float = train_repr.float()
        src_sq = torch.sum(src_float.square(), dim=-1, keepdim=True)
        train_sq = torch.sum(train_repr_float.square(), dim=-1, keepdim=True).transpose(1, 2)
        interaction = torch.matmul(src_float, train_repr_float.transpose(1, 2))
        sq_dist = (src_sq + train_sq - 2.0 * interaction).clamp_min(0.0)
        sim = -sq_dist / (2.0 * (self.soft_kmeans_temperature * torch.sqrt(d))**2)
        log_assign = torch.log_softmax(sim, dim=-1)
        return self._aggregate_kernel_assignments(log_assign, y_train)

    def _euclidean_decoder(
        self, src: Tensor, y_train: Tensor, train_size: int
    ) -> Tensor:
        """Decode using negative Euclidean distance to training representations."""

        if self.max_classes <= 0:
            raise ValueError("euclidean decoder is only supported for classification (max_classes > 0).")
        if train_size <= 0:
            raise ValueError("euclidean decoder requires train_size > 0")

        train_repr = src[:, :train_size, :]
        src_float = src.float()
        train_repr_float = train_repr.float()
        src_sq = torch.sum(src_float.square(), dim=-1, keepdim=True)
        train_sq = torch.sum(train_repr_float.square(), dim=-1, keepdim=True).transpose(1, 2)
        interaction = torch.matmul(src_float, train_repr_float.transpose(1, 2))
        sq_dist = (src_sq + train_sq - 2.0 * interaction).clamp_min(0.0)
        sim = -torch.sqrt(sq_dist.clamp_min(torch.finfo(src_float.dtype).eps))
        log_assign = torch.log_softmax(sim / self.soft_kmeans_temperature, dim=-1)
        return self._aggregate_kernel_assignments(log_assign, y_train)

    def _apply_icl_backbone(self, R: Tensor, train_size: int, y_train: Optional[Tensor] = None) -> Tensor:
        """Apply encoder ICL backbone."""

        if self.icl_backend != "encoder":
            raise ValueError("_apply_icl_backbone is encoder-only; graph backend uses _run_graph_column_pipeline")
        return self.tf_icl(R, train_size=train_size)

    def _grouping(self, num_classes: int) -> tuple[Tensor, int]:
        """Divide classes into balanced groups for hierarchical classification.

        This method implements a balanced partitioning strategy that divides classes
        into approximately equal-sized groups to minimize tree depth. The number of
        groups formed at this level will not exceed `max_classes`.

        Parameters
        ----------
        num_classes : int
            Total number of unique classes to partition into groups.

        Returns
        -------
        group_assignments : Tensor
            Tensor mapping each class index to its assigned group (0-indexed).

        num_groups : int
            Total number of groups created (will be <= max_classes).

        Notes
        -----
        For example, with max_classes=10 and num_classes=25:
        - Distributes 25 classes into 3 groups. Sizes: [9, 8, 8].
        - Returns assignments tensor and num_groups = 3.

        With max_classes=10 and num_classes=101:
        - Distributes 101 classes into 10 groups. Sizes: [11, 10, 10, 10, 10, 10, 10, 10, 10, 10].
        - Returns assignments tensor and num_groups = 10.
        - The child node receiving 11 classes will be further divided into 2 groups: [6, 5].
        """

        if num_classes <= self.max_classes:
            return torch.zeros(num_classes, dtype=torch.int), 1

        num_groups = min(math.ceil(num_classes / self.max_classes), self.max_classes)
        group_assignments = torch.zeros(num_classes, dtype=torch.int)
        current_pos = 0

        remaining_classes = num_classes
        remaining_groups = num_groups
        for i in range(num_groups):
            group_size = math.ceil(remaining_classes / remaining_groups)
            group_assignments[current_pos : current_pos + group_size] = i
            current_pos += group_size
            remaining_classes -= group_size
            remaining_groups -= 1

        return group_assignments, num_groups

    def _fit_node(self, node: ClassNode, R: Tensor, y: Tensor, current_depth: int):
        """Recursively build a node in the hierarchical classification tree.

        For each node, this method either:

        1. Creates a leaf node if the number of classes is small enough to handle directly
        2. Splits classes into groups and recursively creates child nodes for each group

        Parameters
        ----------
        node : ClassNode
            Current node being constructed in the tree.

        R : Tensor
            Row representations of shape (num_samples, D) where num_samples is the number of
            examples assigned to this node.

        y : Tensor
            Targets of shape (num_samples,) corresponding to the samples in R.

        current_depth : int
            Current depth in the hierarchical tree (root = 0).
        """

        unique_classes = torch.unique(y).int()
        node.classes_ = unique_classes

        if len(unique_classes) <= self.max_classes:
            # Create leaf node for direct classification
            node.is_leaf = True
            node.R = R
            node.y = y
            return

        # Merge classes into groups
        group_assignments, num_groups = self._grouping(len(unique_classes))

        # Create mapping from original class labels to their corresponding group numbers
        node.class_mapping = {c.item(): g.item() for c, g in zip(unique_classes, group_assignments)}
        node.group_indices = torch.tensor([node.class_mapping[c.item()] for c in y], dtype=torch.int)
        node.R = R
        node.y = y
        node.is_leaf = False

        # Create child nodes for each group
        for group in range(num_groups):
            mask = node.group_indices == group
            child_node = ClassNode(current_depth + 1)
            self._fit_node(child_node, R[mask], y[mask], current_depth + 1)
            node.child_nodes.append(child_node)

    def _fit_hierarchical(self, R_train: Tensor, y_train: Tensor):
        """Initialize the hierarchical classification tree.

        Parameters
        ----------
        R_train : Tensor
            Row representations of training data of shape (train_size, D).

        y_train : Tensor
            Training targets of shape (train_size,).
        """

        self.root = ClassNode(depth=0)
        self._fit_node(self.root, R_train, y_train, current_depth=0)

    def _label_encoding(self, y: Tensor) -> Tensor:
        """Remapping target values to contiguous integers starting from 0."""

        unique_vals, _ = torch.unique(y, return_inverse=True)
        indices = unique_vals.argsort()
        return indices[torch.searchsorted(unique_vals, y)]

    def _run_graph_column_pipeline(
        self,
        graph_input: Tensor,
        y_train: Tensor,
        graph_set: Optional[CompactGraphSet] = None,
    ) -> Tensor:
        """Graph-only column pipeline.

        Parameters
        ----------
        graph_input : Tensor
            Graph-ready input of shape (B, T, C_total, D), where C_total includes
            appended graph CLS columns and pre-encoded train-label signal.

        y_train : Tensor
            Training labels of shape (B, train_size).

        Returns
        -------
        Tensor
            Sequence representations of shape (B, T, d_model).
        """

        if graph_input.ndim != 4:
            raise ValueError("Graph column pipeline expects graph_input with shape (B, T, C, D)")

        B, T, C, D = graph_input.shape
        if D != self.graph_col_dim:
            raise ValueError(
                f"Graph column pipeline expects last dim {self.graph_col_dim}, got {D}."
            )
        if C < self.graph_num_cls:
            raise ValueError(
                f"Graph column pipeline expects at least {self.graph_num_cls} columns for CLS outputs, got {C}."
            )

        if graph_set is None:
            raise ValueError("Graph backend requires precomputed graph metadata")

        if graph_set.num_graphs != self.graph_num_graphs:
            raise ValueError(
                f"Expected {self.graph_num_graphs} graph batches, got {graph_set.num_graphs}"
            )
        if graph_set.num_nodes != T:
            raise ValueError(f"Expected graph num_nodes={T}, got {graph_set.num_nodes}")

        if graph_set.num_datasets != B:
            raise ValueError(f"Expected {B} graph datasets, got {graph_set.num_datasets}")
        if graph_set.edge_index.ndim != 2 or graph_set.edge_index.shape[0] != 2:
            raise ValueError("Compact graph edge_index must have shape (2, E)")
        if graph_set.edge_offsets.ndim != 2:
            raise ValueError("Compact graph edge_offsets must have shape (num_graphs, num_datasets + 1)")

        x = graph_input
        return self.gat_icl(x, graph_set=graph_set)

    def prepare_graph_input(
        self,
        col_embeddings: Tensor,
        y_train: Tensor,
        pre_col_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """Prepare graph-ready 4D input with column identity and train-label signals.

        Parameters
        ----------
        col_embeddings : Tensor
            Column embeddings of shape (B, T, C, D).

        y_train : Tensor
            Training labels of shape (B, train_size).

        pre_col_embeddings : Optional[Tensor], default=None
            Optional projected input features of shape (B, T, C, D) carrying
            column-identity signal.

        Returns
        -------
        Tensor
            Graph-ready tensor of shape (B, T, C + graph_num_cls, D).
        """

        if self.icl_backend != "graph":
            raise ValueError("prepare_graph_input is supported only for graph ICL backend")
        if col_embeddings.ndim != 4:
            raise ValueError("col_embeddings must have shape (B, T, C, D)")

        B, _, _, D = col_embeddings.shape
        if D != self.graph_col_dim:
            raise ValueError(f"Expected last dim {self.graph_col_dim}, got {D}")

        x = col_embeddings.clone()
        if pre_col_embeddings is not None:
            if pre_col_embeddings.shape != col_embeddings.shape:
                raise ValueError("pre_col_embeddings must have the same shape as col_embeddings")
            x = x + pre_col_embeddings

        train_size = y_train.shape[1]
        y_encoded = self.y_encoder(y_train.float()).view(B, train_size, self.graph_num_cls, D)
        x[:, :train_size, -self.graph_num_cls :, :] = x[:, :train_size, -self.graph_num_cls :, :] + y_encoded

        cls_tokens = self.graph_cls_tokens.view(1, 1, self.graph_num_cls, D).expand(B, x.shape[1], -1, -1)
        x = torch.cat([x, cls_tokens], dim=2)
        
        return x

    def _icl_predictions(
        self,
        R: Tensor,
        y_train: Tensor,
        return_pre_decoder_repr: bool = False,
        use_col_embeddings: bool = False,
        pre_col_embeddings: Optional[Tensor] = None,
        graph_set: Optional[SparseGraphSet] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """In-context learning predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        Returns
        -------
        Tensor
            Predictions of shape (B, T, out_dim):

            - For regression (max_classes=0): out_dim = num_quantiles
            - For classification (max_classes>0): out_dim = max_classes
        """

        train_size = y_train.shape[1]
        if self.icl_backend == "graph":
            if R.ndim != 4:
                raise ValueError(
                    "Graph backend expects R with shape (B, T, C, D). "
                    "Use prepare_graph_input for graph-mode inputs."
                )
            src = self._run_graph_column_pipeline(R, y_train=y_train, graph_set=graph_set)
        else:
            if self.max_classes > 0:  # Classification
                Ry_train = self.y_encoder(y_train.float())
            else:  # Regression
                Ry_train = self.y_encoder(y_train.unsqueeze(-1))

            R[:, :train_size] = R[:, :train_size] + Ry_train
            src = self._apply_icl_backbone(R, train_size=train_size, y_train=y_train)
        if self.norm_first:
            src = self.ln(src)
        if self.decoder_type == "mlp":
            out = self.decoder(src)
        elif self.decoder_type == "soft_kmeans":
            out = self._soft_kmeans_decoder(src, y_train=y_train, train_size=train_size)
        elif self.decoder_type == "rbf":
            out = self._rbf_decoder(src, y_train=y_train, train_size=train_size)
        else:
            out = self._euclidean_decoder(src, y_train=y_train, train_size=train_size)

        if return_pre_decoder_repr:
            return out, src

        return out

    def _predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = False,
        softmax_temperature: float = 0.9,
        auto_batch: bool = True,
    ) -> Tensor:
        """Generate predictions for standard classification with up to `max_classes` classes.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        return_logits : bool, default=False
            If True, return logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        auto_batch : bool, default=True
            Whether to use InferenceManager to automatically split inputs into smaller batches.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles), where test_size = T - train_size

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        out = self.inference_mgr(
            self._icl_predictions, inputs=OrderedDict([("R", R), ("y_train", y_train)]), auto_batch=auto_batch
        )

        train_size = y_train.shape[1]
        if self.max_classes == 0:
            out = out[:, train_size:]
        else:
            num_classes = len(torch.unique(y_train[0]))
            out = out[:, train_size:, :num_classes]
            if self.decoder_type in ("soft_kmeans", "rbf", "euclidean"):
                if not return_logits:
                    out = out.exp()
            elif not return_logits:
                out = torch.softmax(out / softmax_temperature, dim=-1)

        return out

    def _predict_hierarchical(
        self, R_test: Tensor, softmax_temperature: float = 0.9, inference_recurrence: Optional[int] = None
    ) -> Tensor:
        """Generate predictions using the hierarchical classification tree.

        This method traverses the tree from leaves to root, computing probabilities at each level
        and combining them according to the probability chain rule.

        Parameters
        ----------
        R_test : Tensor
            Row representations of test data of shape (test_size, D).

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        Returns
        -------
        Tensor
            Probability over all classes, shape (test_size, C).
        """

        test_size = R_test.shape[0]
        device = R_test.device
        num_classes = len(self.root.classes_)

        def process_node(node, R_test):
            """Recursively process a node in the hierarchical tree.

            For leaf nodes: Directly predict class probabilities within the node's subset
            For internal nodes: Combine predictions from child nodes weighted by group probabilities
            """

            # Concatenate test data with node data
            node_R = torch.cat([node.R.to(device), R_test], dim=0)

            # Case 1: Leaf node - direct classification
            if node.is_leaf:
                node_y = self._label_encoding(node.y.to(device))
                # Get predictions for this leaf
                leaf_preds = self._predict_standard(
                    R=node_R.unsqueeze(0),
                    y_train=node_y.unsqueeze(0),
                    softmax_temperature=softmax_temperature,
                    auto_batch=False,
                ).squeeze(0)
                # Map leaf predictions to the global class space
                global_preds = torch.zeros((test_size, num_classes), device=device)
                for local_idx, global_idx in enumerate(node.classes_):
                    global_preds[:, global_idx] = leaf_preds[:, local_idx]

                return global_preds

            # Case 2: Internal node - classification into groups
            # Initialize output tensor for all classes
            final_probs = torch.zeros((test_size, num_classes), device=device)

            # Get group probabilities for this node
            node_y = node.group_indices.to(device)
            group_probs = self._predict_standard(
                R=node_R.unsqueeze(0),
                y_train=node_y.unsqueeze(0),
                softmax_temperature=softmax_temperature,
                auto_batch=False,
            ).squeeze(0)

            # Recursively process child nodes and combine predictions
            for group_idx, child_node in enumerate(node.child_nodes):
                child_probs = process_node(child_node, R_test)
                final_probs += child_probs * group_probs[:, group_idx : group_idx + 1]

            return final_probs

        return process_node(self.root, R_test)

    def _inference_forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """In-context learning based on learned row representations for inference.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        return_logits : bool, default=True
            If True, return logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles), where test_size = T - train_size

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        if self.max_classes == 0:  # Regression
            out = self._predict_standard(R, y_train)
        else:  # Classification
            num_classes = len(torch.unique(y_train[0]))
            assert all(
                len(torch.unique(yi)) == num_classes for yi in y_train
            ), "All tables must have the same number of classes"

            if num_classes <= self.max_classes:
                # Standard classification
                out = self._predict_standard(
                    R, y_train, return_logits=return_logits, softmax_temperature=softmax_temperature
                )
            else:
                # Hierarchical classification
                out = []
                train_size = y_train.shape[1]
                for ri, yi in zip(R, y_train):
                    if mgr_config.offload:
                        ri, yi = ri.cpu(), yi.cpu()
                    else:
                        ri, yi = ri.to(mgr_config.device), yi.to(mgr_config.device)
                    self._fit_hierarchical(ri[:train_size], yi)
                    probs = self._predict_hierarchical(ri[train_size:])
                    out.append(probs)
                out = torch.stack(out, dim=0)
                if return_logits:
                    out = softmax_temperature * torch.log(out + 1e-6)

        return out

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
        return_pre_decoder_repr: bool = False,
        use_col_embeddings: bool = False,
        pre_col_embeddings: Optional[Tensor] = None,
        graph_set: Optional[SparseGraphSet] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """In-context learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        return_logits : bool, default=True
            If True, return logits instead of probabilities. Used only in inference mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in inference mode.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

        Returns
        -------
        Tensor
            For training mode:
                Predictions of shape (B, test_size, out_dim):

                - For regression (max_classes=0): out_dim = num_quantiles
                - For classification (max_classes>0): out_dim = max_classes

            For inference mode:
                For regression (max_classes=0):
                    Predictions of shape (B, test_size, num_quantiles)

                For classification (max_classes>0):
                    If return_logits=True: Logits of shape (B, test_size, num_classes)
                    If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if self.training:
            train_size = y_train.shape[1]
            if return_pre_decoder_repr:
                out, pre_decoder_repr = self._icl_predictions(
                    R,
                    y_train,
                    return_pre_decoder_repr=True,
                    graph_set=graph_set,
                )
                out = out[:, train_size:]
                pre_decoder_repr = pre_decoder_repr[:, train_size:]
                return out, pre_decoder_repr

            out = self._icl_predictions(R, y_train, graph_set=graph_set)
            out = out[:, train_size:]
        else:
            if self.icl_backend == "graph":
                out = self._icl_predictions(R, y_train, graph_set=graph_set)
                train_size = y_train.shape[1]
                out = out[:, train_size:]
                if self.max_classes > 0:
                    num_classes = len(torch.unique(y_train[0]))
                    out = out[..., :num_classes]
                    if self.decoder_type in ("soft_kmeans", "rbf", "euclidean"):
                        if not return_logits:
                            out = out.exp()
                    elif not return_logits:
                        out = torch.softmax(out / softmax_temperature, dim=-1)
            else:
                out = self._inference_forward(R, y_train, return_logits, softmax_temperature, mgr_config)

        return out

    def prepare_repr_cache(self, R: Tensor, y_train: Tensor) -> Tensor:
        """Add target embedding to train representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        Returns
        -------
        Tensor
            Full representations with y_train baked into the train portion,
            shape (B, T, D).
        """

        train_size = y_train.shape[1]
        if self.max_classes > 0:
            Ry_train = self.y_encoder(y_train.float())
        else:
            Ry_train = self.y_encoder(y_train.unsqueeze(-1))
        R[:, :train_size] = R[:, :train_size] + Ry_train

        return R

    def _icl_predictions_repr_cache(self, R: Tensor, train_size: int) -> Tensor:
        """In-context learning predictions with representation cache.

        This method does not add target embedding because it is already
        baked into the cached train representations.

        Parameters
        ----------
        R : Tensor
            Full representations of shape (B, T, D) where
            R[:, :train_size] has y_train already baked in.

        train_size : int
            Number of training samples.

        Returns
        -------
        Tensor
            Predictions of shape (B, T, out_dim).
        """

        if self.icl_backend == "graph":
            raise ValueError("Representation-cache path is not supported for graph ICL backend.")
        if self.decoder_type != "mlp":
            raise ValueError("Representation-cache path currently supports decoder_type='mlp' only.")

        src = self._apply_icl_backbone(R, train_size=train_size)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)

        return out

    def forward_with_repr_cache(
        self,
        R: Tensor,
        train_size: int,
        num_classes: Optional[int] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """In-context learning with representation cache.

        Runs the ICL transformer on pre-assembled representations where
        the training portion already has y_train baked in.

        Parameters
        ----------
        R : Tensor
            Full representations of shape (B, T, D) where
            R[:, :train_size] has y_train already baked in.

        train_size : int
            Number of training samples.

        num_classes : Optional[int], default=None
            Number of classes for classification tasks.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. If None, uses the default
            ICL_CONFIG from InferenceConfig.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles)

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if self.icl_backend == "graph":
            raise ValueError("Representation-cache path is not supported for graph ICL backend.")

        if mgr_config is None:
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        out = self.inference_mgr(
            self._icl_predictions_repr_cache,
            inputs=OrderedDict([("R", R), ("train_size", train_size)]),
        )

        out = out[:, train_size:]
        if self.max_classes > 0:
            assert num_classes is not None, "num_classes must be provided for classification"
            out = out[..., :num_classes]
            if not return_logits:
                out = torch.softmax(out / softmax_temperature, dim=-1)

        return out

    def _icl_predictions_with_cache(
        self,
        R: Tensor,
        icl_cache: KVCache,
        y_train: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """In-context learning predictions with KV caching.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D).

        icl_cache : KVCache
            Cache object for storing/retrieving K/V projections.

        y_train : Optional[Tensor], default=None
            Training targets of shape (B, train_size). Required when store_cache=True;
            ignored when use_cache=True.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        Returns
        -------
        Tensor
            Predictions of shape (B, T, out_dim) or (B, test_size, out_dim) when use_cache=True:

            - For regression (max_classes=0): out_dim = num_quantiles
            - For classification (max_classes>0): out_dim = max_classes
        """
        if self.icl_backend == "graph":
            raise ValueError("KV cache path is not supported for graph ICL backend.")
        if self.decoder_type != "mlp":
            raise ValueError("KV cache path currently supports decoder_type='mlp' only.")

        # When using cache, skip y_train embedding — it's already baked
        # into the cached K/V projections from the store_cache pass.
        if store_cache:
            assert y_train is not None, "y_train must be provided when store_cache=True"
            train_size = y_train.shape[1]

            if self.max_classes > 0:  # Classification
                Ry_train = self.y_encoder(y_train.float())
            else:  # Regression
                Ry_train = self.y_encoder(y_train.unsqueeze(-1))
            R[:, :train_size] = R[:, :train_size] + Ry_train

        src = self.tf_icl.forward_with_cache(
            R,
            icl_cache=icl_cache,
            train_size=train_size if store_cache else None,
            use_cache=use_cache,
            store_cache=store_cache,
        )
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)

        return out

    def forward_with_cache(
        self,
        R: Tensor,
        icl_cache: KVCache,
        y_train: Optional[Tensor] = None,
        pre_col_embeddings: Optional[Tensor] = None,
        num_classes: Optional[int] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        use_cache: bool = False,
        store_cache: bool = True,
        mgr_config: MgrConfig = None,
        use_col_embeddings: bool = False,
    ) -> Tensor:
        """In-context learning with KV caching support.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D).

        icl_cache : KVCache
            Cache object for storing/retrieving K/V projections.

        y_train : Optional[Tensor], default=None
            Training targets of shape (B, train_size). Required when store_cache=True;
            ignored when use_cache=True.

        num_classes : Optional[int], default=None
            Number of classes for classification. If None, computed from y_train.
            When use_cache=True, this should be provided from the cache.

        return_logits : bool, default=True
            If True, return logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. If None, uses the default
            ICL_CONFIG from InferenceConfig.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles)

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if self.icl_backend == "graph":
            raise ValueError("KV cache path is not supported for graph ICL backend.")

        if use_col_embeddings:
            raise NotImplementedError(
                "Cache path for graph column-embedding pipeline is not implemented yet."
            )

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache:
            assert y_train is not None, "y_train must be provided when store_cache=True"
            # many-class classification is not supported with caching
            if self.max_classes > 0:
                num_classes = len(torch.unique(y_train[0]))
                if num_classes > self.max_classes:
                    raise ValueError(
                        f"KV caching is not supported for classification with more classes "
                        f"({num_classes}) than max_classes ({self.max_classes}). Hierarchical classification "
                        f"requires multiple forward passes which is incompatible with caching."
                    )
        else:
            assert num_classes is not None, "num_classes must be provided when use_cache=True"

        if mgr_config is None:
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        out = self.inference_mgr(
            self._icl_predictions_with_cache,
            inputs=OrderedDict(
                [
                    ("R", R),
                    ("icl_cache", icl_cache),
                    ("y_train", y_train),
                    ("use_cache", use_cache),
                    ("store_cache", store_cache),
                ]
            ),
        )

        if store_cache:
            train_size = y_train.shape[1]
            out = out[:, train_size:]

        if self.max_classes > 0:
            out = out[..., :num_classes]
            if not return_logits:
                out = torch.softmax(out / softmax_temperature, dim=-1)

        return out
