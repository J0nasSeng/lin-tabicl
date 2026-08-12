import torch
import pytest

from src.tabicl._model.graph import (
    GraphPrior,
    SparseGraphBatch,
    SparseGraphSet,
    build_class_conditioned_graph,
    build_class_conditioned_graphs,
    induce_graph_set,
    stack_graph_sets,
)
from src.tabicl._model.gat import GraphMultiheadAttention, GraphAttentionTransformer
from src.tabicl._model.gat import GraphAttentionBlock
from src.tabicl._model.learning import ICLearning
from src.tabicl._model.tabicl import TabICL
from src.tabicl.prior._dataset import PriorDataset
from src.tabicl.train._losses import entropy_regularizer


def _build_labels(batch_size: int, train_size: int, num_classes: int) -> torch.Tensor:
    labels = []
    per_class = train_size // num_classes
    for _ in range(batch_size):
        y = []
        for c in range(num_classes):
            y.extend([c] * per_class)
        labels.append(torch.tensor(y, dtype=torch.long))
    return torch.stack(labels, dim=0)


def test_graph_builder_train_degree_and_test_class_coverage():
    batch_size = 1
    train_size = 15
    test_size = 5
    total_nodes = train_size + test_size
    num_classes = 3

    y_train = _build_labels(batch_size=batch_size, train_size=train_size, num_classes=num_classes)

    graph = build_class_conditioned_graph(
        y_train=y_train,
        total_nodes=total_nodes,
        min_train_neighbors=8,
        max_train_neighbors=15,
        cross_label_fraction=0.1,
        train_neighbors_per_test=3,
        seed=7,
    )

    edge_index = graph.edge_index[0]
    src = edge_index[0].long()
    dst = edge_index[1].long()

    # Graph generation should not include self-connections.
    assert int((src == dst).sum().item()) == 0

    # Train/train non-self edges should remain bidirectional.
    edge_set = set(zip(src.tolist(), dst.tolist()))
    for u, v in edge_set:
        if u != v and u < train_size and v < train_size:
            assert (v, u) in edge_set

    # Test nodes are consumers only: no test->train edges.
    test_to_train = (src >= train_size) & (dst < train_size)
    assert int(test_to_train.sum().item()) == 0

    # Same-label edges should be present within every class among train nodes.
    labels = y_train[0]
    classes = torch.unique(labels)
    for c in classes:
        class_nodes = torch.where(labels == c)[0]
        class_mask = (src < train_size) & (dst < train_size)
        class_mask &= torch.isin(src, class_nodes) & torch.isin(dst, class_nodes)
        assert int(class_mask.sum().item()) > 0

    # Each test node should connect to at least 3 train nodes from each class.
    for test_node in range(train_size, total_nodes):
        for c in classes:
            class_mask = labels[src.clamp_max(train_size - 1)] == c
            incoming = (dst == test_node) & (src < train_size)
            class_incoming = incoming & class_mask
            assert int(class_incoming.sum().item()) >= 3


def test_graph_builder_multiple_graphs_are_reproducible_and_independent():
    y_train = _build_labels(batch_size=1, train_size=15, num_classes=3)

    graph_set_a = build_class_conditioned_graphs(
        y_train=y_train,
        total_nodes=20,
        num_graphs=3,
        min_train_neighbors=1,
        max_train_neighbors=3,
        train_neighbors_per_test=2,
        seed=11,
    )
    graph_set_b = build_class_conditioned_graphs(
        y_train=y_train,
        total_nodes=20,
        num_graphs=3,
        min_train_neighbors=1,
        max_train_neighbors=3,
        train_neighbors_per_test=2,
        seed=11,
    )

    assert graph_set_a.num_graphs == 3
    for graph_a, graph_b in zip(graph_set_a.graphs, graph_set_b.graphs):
        assert torch.equal(graph_a.edge_index[0], graph_b.edge_index[0])
    assert not torch.equal(
        graph_set_a.graphs[0].edge_index[0], graph_set_a.graphs[1].edge_index[0]
    )


def test_graph_prior_graph_tasks_reuse_one_topology_across_gat_slots():
    labels = _build_labels(batch_size=2, train_size=15, num_classes=3)
    graph_set = GraphPrior(
        graph_v1_prob=0.0,
        graph_v2_prob=0.0,
        graph_prob=1.0,
        min_train_neighbors=1,
        max_train_neighbors=3,
        train_neighbors_per_test=2,
        seed=11,
    )(labels, n_train=15, num_graphs=3)

    assert graph_set.num_graphs == 3
    for dataset_idx in range(labels.shape[0]):
        reference = graph_set.graphs[0].edge_index[dataset_idx]
        for graph_idx in range(1, graph_set.num_graphs):
            assert torch.equal(reference, graph_set.graphs[graph_idx].edge_index[dataset_idx])


def test_induce_graph_set_preserves_topology_and_all_test_nodes():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long
    )
    graph_set = SparseGraphSet([SparseGraphBatch(edge_index=[edge_index], num_nodes=6)])

    induced = induce_graph_set(
        graph_set,
        train_mask=torch.tensor([True, False, True]),
        train_size=3,
    )

    assert induced.num_nodes == 5
    # Original vertices [0, 2, 3, 4, 5] map to [0, 1, 2, 3, 4].
    assert torch.equal(
        induced.graphs[0].edge_index[0],
        torch.tensor([[1, 2, 3, 4], [2, 1, 4, 3]]),
    )


def test_graph_builder_fraction_controls_unique_train_train_edges():
    y_train = _build_labels(batch_size=1, train_size=12, num_classes=3)

    def count_train_edges(cross_label_fraction: float) -> tuple[int, int]:
        graph = build_class_conditioned_graph(
            y_train=y_train,
            total_nodes=12,
            min_train_neighbors=1,
            max_train_neighbors=1,
            cross_label_fraction=cross_label_fraction,
            train_neighbors_per_test=1,
            seed=17,
        )
        edge_index = graph.edge_index[0]
        src, dst = edge_index.long()
        labels = y_train[0]
        train_edges = (src < 12) & (dst < 12)
        same = train_edges & (labels[src.long()] == labels[dst.long()])
        return int(same.sum().item()), int((train_edges & ~same).sum().item())

    same_only = count_train_edges(0.0)
    cross_only = count_train_edges(1.0)
    balanced = count_train_edges(0.5)

    assert same_only[0] > 0 and same_only[1] == 0
    assert cross_only[0] == 0 and cross_only[1] > 0
    assert balanced[0] == balanced[1]


def test_graph_attention_transformer_routes_graphs_by_layer_group():
    transformer = GraphAttentionTransformer(
        num_blocks=4,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        num_graphs=2,
    )
    assert transformer.num_graphs == 2
    assert transformer.layers_per_graph == 2

    with pytest.raises(ValueError, match="divisible"):
        GraphAttentionTransformer(
            num_blocks=3,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            num_graphs=2,
        )

    with pytest.raises(ValueError, match="Expected 2 graph batches"):
        transformer(torch.randn(1, 4, 2, 8), [[]])


def test_prior_dataset_samples_graphs_with_each_dataset():
    prior = PriorDataset(
        batch_size=2,
        batch_size_per_gp=2,
        min_features=2,
        max_features=2,
        max_classes=3,
        min_seq_len=8,
        max_seq_len=9,
        min_train_size=0.4,
        max_train_size=0.6,
        prior_type="dummy",
        graph_num_graphs=2,
        graph_min_train_neighbors=1,
        graph_max_train_neighbors=2,
        graph_train_neighbors_per_test=1,
    )

    batch = prior.get_batch()
    assert len(batch) == 6
    graph_sets = batch[-1]
    assert len(graph_sets) == 2
    assert all(graph_set.num_graphs == 2 for graph_set in graph_sets)
    stacked = stack_graph_sets(graph_sets)
    assert len(stacked.graphs[0].edge_index) == 2


def test_iclearning_graph_backend_forward_shape():
    batch_size = 2
    train_size = 12
    total_nodes = 20
    d_model = 16

    y_train = _build_labels(batch_size=batch_size, train_size=train_size, num_classes=3).float()

    model = ICLearning(
        max_classes=5,
        out_dim=5,
        d_model=d_model,
        num_blocks=2,
        nhead=4,
        dim_feedforward=32,
        icl_backend="graph",
        graph_min_train_neighbors=8,
        graph_max_train_neighbors=10,
        graph_cross_label_fraction=0.1,
        graph_train_neighbors_per_test=3,
        graph_seed=0,
    )
    model.train()

    col_embeddings = torch.randn(batch_size, total_nodes, model.graph_num_cls, model.graph_col_dim)
    graph_input = model.prepare_graph_input(col_embeddings=col_embeddings, y_train=y_train)
    graph_set = build_class_conditioned_graphs(
        y_train=y_train.long(),
        total_nodes=total_nodes,
        num_graphs=model.graph_num_graphs,
        min_train_neighbors=model.graph_min_train_neighbors,
        max_train_neighbors=model.graph_max_train_neighbors,
        cross_label_fraction=model.graph_cross_label_fraction,
        train_neighbors_per_test=model.graph_train_neighbors_per_test,
        seed=model.graph_seed,
    )
    out = model(graph_input, y_train, graph_set=graph_set)
    assert out.shape == (batch_size, total_nodes - train_size, 5)
    assert torch.isfinite(out).all()


def test_iclearning_graph_backend_requires_4d_graph_input():
    batch_size = 2
    train_size = 12
    total_nodes = 20
    d_model = 16

    y_train = _build_labels(batch_size=batch_size, train_size=train_size, num_classes=3).float()
    row_repr = torch.randn(batch_size, total_nodes, d_model)

    model = ICLearning(
        max_classes=5,
        out_dim=5,
        d_model=d_model,
        num_blocks=2,
        nhead=4,
        dim_feedforward=32,
        icl_backend="graph",
    )
    model.train()

    with pytest.raises(ValueError, match=r"Graph backend expects R with shape \(B, T, C, D\)"):
        model(row_repr, y_train)


def test_iclearning_graph_backend_hierarchical_soft_kmeans_uses_local_labels():
    batch_size = 2
    train_size = 12
    test_size = 3
    num_classes = 7
    d_model = 12

    y_train = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]] * batch_size,
        dtype=torch.long,
    )
    model = ICLearning(
        max_classes=3,
        out_dim=3,
        d_model=d_model,
        num_blocks=2,
        nhead=3,
        dim_feedforward=24,
        icl_backend="graph",
        decoder_type="soft_kmeans",
        graph_min_train_neighbors=1,
        graph_max_train_neighbors=2,
        graph_train_neighbors_per_test=1,
        graph_seed=0,
    ).eval()

    base_col_embeddings = torch.randn(
        batch_size, train_size + test_size, model.graph_num_cls, model.graph_col_dim
    )
    pre_col_embeddings = torch.randn_like(base_col_embeddings)
    graph_input = model.prepare_graph_input(
        base_col_embeddings,
        y_train,
        pre_col_embeddings=pre_col_embeddings,
        encode_labels=False,
    )
    graph_set = build_class_conditioned_graphs(
        y_train=y_train,
        total_nodes=train_size + test_size,
        num_graphs=model.graph_num_graphs,
        min_train_neighbors=model.graph_min_train_neighbors,
        max_train_neighbors=model.graph_max_train_neighbors,
        train_neighbors_per_test=model.graph_train_neighbors_per_test,
        seed=model.graph_seed,
    )

    out = model(
        graph_input,
        y_train,
        return_logits=False,
        graph_set=graph_set,
        base_col_embeddings=base_col_embeddings,
        pre_col_embeddings=pre_col_embeddings,
    )

    assert out.shape == (batch_size, test_size, num_classes)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(batch_size, test_size), atol=1e-4)


def test_iclearning_graph_checkpoint_state_dict_keys_are_unchanged():
    kwargs = dict(
        max_classes=3,
        out_dim=3,
        d_model=12,
        num_blocks=2,
        nhead=3,
        dim_feedforward=24,
        icl_backend="graph",
        decoder_type="soft_kmeans",
    )
    model_before = ICLearning(**kwargs)
    state_dict = model_before.state_dict()
    model_after = ICLearning(**kwargs)
    missing, unexpected = model_after.load_state_dict(state_dict, strict=False)
    assert missing == []
    assert unexpected == []
    assert set(model_before.state_dict()) == set(model_after.state_dict())


def test_iclearning_graph_backend_rejects_regression():
    with pytest.raises(ValueError, match="classification only"):
        ICLearning(
            max_classes=0,
            out_dim=10,
            d_model=16,
            num_blocks=1,
            nhead=4,
            dim_feedforward=32,
            icl_backend="graph",
        )


def test_iclearning_graph_backend_cache_paths_raise():
    batch_size = 1
    train_size = 9
    total_nodes = 12
    d_model = 16

    y_train = _build_labels(batch_size=batch_size, train_size=train_size, num_classes=3).float()
    R = torch.randn(batch_size, total_nodes, d_model)

    model = ICLearning(
        max_classes=5,
        out_dim=5,
        d_model=d_model,
        num_blocks=1,
        nhead=4,
        dim_feedforward=32,
        icl_backend="graph",
    )

    with pytest.raises(ValueError, match="Representation-cache path"):
        model.forward_with_repr_cache(R, train_size=train_size, num_classes=3)


def test_entropy_regularizer_prefers_uniform_predictions():
    uniform_logits = torch.zeros(8, 5)
    peaked_logits = torch.full((8, 5), -6.0)
    peaked_logits[:, 0] = 6.0

    h_uniform = entropy_regularizer(uniform_logits)
    h_peaked = entropy_regularizer(peaked_logits)

    assert h_uniform > h_peaked
    assert torch.isfinite(h_uniform)
    assert torch.isfinite(h_peaked)


def test_iclearning_soft_kmeans_decoder_forward_shape():
    batch_size = 2
    train_size = 12
    total_nodes = 20
    d_model = 16

    y_train = _build_labels(batch_size=batch_size, train_size=train_size, num_classes=3).long()
    R = torch.randn(batch_size, total_nodes, d_model)

    model = ICLearning(
        max_classes=5,
        out_dim=5,
        d_model=d_model,
        num_blocks=2,
        nhead=4,
        dim_feedforward=32,
        icl_backend="encoder",
        decoder_type="soft_kmeans",
        soft_kmeans_temperature=0.2,
    )
    model.train()

    out, pre = model(R, y_train, return_pre_decoder_repr=True)
    assert out.shape == (batch_size, total_nodes - train_size, 5)
    assert pre.shape == (batch_size, total_nodes - train_size, d_model)
    assert torch.isfinite(out).all()
    assert torch.isfinite(pre).all()


def test_iclearning_rbf_decoder_matches_reference_and_normalizes():
    torch.manual_seed(3)
    batch_size = 2
    train_size = 6
    total_nodes = 9
    max_classes = 5
    temperature = 0.7
    y_train = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [2, 0, 1, 2, 0, 1]], dtype=torch.long
    )
    src = torch.randn(batch_size, total_nodes, 8, requires_grad=True)
    model = ICLearning(
        max_classes=max_classes,
        out_dim=max_classes,
        d_model=8,
        num_blocks=1,
        nhead=2,
        dim_feedforward=16,
        icl_backend="encoder",
        decoder_type="rbf",
        soft_kmeans_temperature=temperature,
    )

    log_probs = model._rbf_decoder(src, y_train, train_size)
    probs = log_probs.exp()
    assert log_probs.shape == (batch_size, total_nodes, max_classes)
    assert torch.isfinite(log_probs).all()
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, total_nodes), atol=1e-6)

    src_float = src.float()
    train = src_float[:, :train_size]
    sq_dist = (
        src_float.square().sum(dim=-1, keepdim=True)
        + train.square().sum(dim=-1, keepdim=True).transpose(1, 2)
        - 2 * torch.matmul(src_float, train.transpose(1, 2))
    ).clamp_min(0)
    assign = torch.softmax(-sq_dist / (2 * (temperature * src.shape[-1] ** 0.5) ** 2), dim=-1)
    reference = torch.zeros(batch_size, total_nodes, max_classes)
    for batch_index in range(batch_size):
        for class_index in range(max_classes):
            class_rows = y_train[batch_index] == class_index
            reference[batch_index, :, class_index] = assign[batch_index, :, class_rows].sum(dim=-1)
    assert torch.allclose(probs, reference, atol=1e-6)

    (-log_probs[:, train_size:].gather(-1, y_train[:, : total_nodes - train_size].unsqueeze(-1))).mean().backward()
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()


def test_iclearning_rbf_decoder_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="soft_kmeans_temperature must be > 0"):
        ICLearning(
            max_classes=3,
            out_dim=3,
            d_model=8,
            num_blocks=1,
            nhead=2,
            dim_feedforward=16,
            icl_backend="encoder",
            decoder_type="rbf",
            soft_kmeans_temperature=0,
        )


def test_iclearning_euclidean_decoder_matches_reference():
    torch.manual_seed(4)
    y_train = torch.tensor([[0, 1, 0, 2]], dtype=torch.long)
    src = torch.randn(1, 6, 8, requires_grad=True)
    model = ICLearning(
        max_classes=4,
        out_dim=4,
        d_model=8,
        num_blocks=1,
        nhead=2,
        dim_feedforward=16,
        icl_backend="encoder",
        decoder_type="euclidean",
    )

    log_probabilities = model._euclidean_decoder(src, y_train, train_size=4)
    probabilities = log_probabilities.exp()
    src_normalized = torch.nn.functional.normalize(src.float(), p=2, dim=-1)
    train_normalized = torch.nn.functional.normalize(src[:, :4].float(), p=2, dim=-1)
    distances = torch.cdist(src_normalized, train_normalized, p=2).clamp_min(
        torch.sqrt(torch.finfo(torch.float32).eps)
    )
    assignments = torch.softmax(-distances / model.soft_kmeans_temperature, dim=-1)
    reference = torch.zeros_like(probabilities)
    for class_index in range(model.max_classes):
        reference[:, :, class_index] = assignments[:, :, y_train[0] == class_index].sum(dim=-1)

    assert torch.allclose(probabilities, reference, atol=1e-6)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1, 6), atol=1e-6)
    (-log_probabilities[:, 4:, 0]).mean().backward()
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()


def test_graph_builder_shared_graph_when_labels_identical():
    batch_size = 3
    train_size = 15
    total_nodes = 22
    y_one = _build_labels(batch_size=1, train_size=train_size, num_classes=3)
    y_train = y_one.expand(batch_size, -1).clone()

    graph_set = build_class_conditioned_graphs(
        y_train=y_train,
        total_nodes=total_nodes,
        num_graphs=1,
        seed=17,
        share_graph_across_batch=True,
    )
    graph = graph_set.graphs[0]

    assert len(graph.edge_index) == batch_size
    for idx in range(1, batch_size):
        assert torch.equal(graph.edge_index[0], graph.edge_index[idx])


def test_graph_builder_shared_graph_fallback_for_non_identical_labels():
    y_train = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 1, 2, 2],
        ],
        dtype=torch.long,
    )
    total_nodes = 10

    graph_set = build_class_conditioned_graphs(
        y_train=y_train,
        total_nodes=total_nodes,
        num_graphs=1,
        seed=3,
        share_graph_across_batch=True,
    )
    graph = graph_set.graphs[0]

    assert len(graph.edge_index) == y_train.shape[0]
    assert not torch.equal(graph.edge_index[0], graph.edge_index[1])


def test_graph_multihead_attention_vectorized_matches_legacy_loop():
    torch.manual_seed(0)
    B, T, C, D, H = 3, 7, 1, 16, 4

    model = GraphMultiheadAttention(d_model=D, nhead=H, dropout=0.0, learnable_residual=True)
    model.eval()

    src = torch.randn(B, T, C, D)
    edge_index_batch = [
        torch.tensor([[0, 1, 2, 2], [1, 2, 0, 3]], dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
        torch.tensor([[1, 3, 4, 5, 6], [0, 2, 2, 6, 1]], dtype=torch.long),
    ]

    with torch.no_grad():
        out_new = model(src, edge_index_batch)
        alpha = torch.sigmoid(model.alpha)

        out_old = []
        for b in range(B):
            x = src[b]
            edge_index = edge_index_batch[b].to(device=x.device, dtype=torch.long)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]

            if edge_src.numel() == 0:
                out_old.append((1.0 - alpha) * x)
                continue

            q = model.q_proj(x).view(T, C, model.nhead, model.head_dim)
            k = model.k_proj(x).view(T, C, model.nhead, model.head_dim)
            v = model.v_proj(x).view(T, C, model.nhead, model.head_dim)

            q_dst = q[edge_dst]
            k_src = k[edge_src]
            v_src = v[edge_src]

            attn_logits = (q_dst * k_src).sum(dim=-1) * model.scale

            edge_weight = torch.zeros_like(attn_logits)
            unique_dst = torch.unique(edge_dst)
            unique_col = torch.arange(C, device=src.device, dtype=torch.long)
            for h in range(model.nhead):
                for c in unique_col.tolist():
                    logits_hc = attn_logits[:, c, h]
                    for dst_idx in unique_dst.tolist():
                        mask = edge_dst == dst_idx
                        if mask.any():
                            edge_weight[mask, c, h] = torch.softmax(logits_hc[mask], dim=0)

            messages = v_src * edge_weight.unsqueeze(-1)

            agg = torch.zeros((T, C, model.nhead, model.head_dim), dtype=src.dtype, device=src.device)
            for c in unique_col.tolist():
                for h in range(model.nhead):
                    agg[:, c, h, :].index_add_(0, edge_dst, messages[:, c, h, :])

            attn_out = model.out_proj(agg.reshape(T, C, model.d_model))
            out_old.append((1.0 - alpha) * x + alpha * attn_out)

        out_old = torch.stack(out_old, dim=0)

    assert torch.allclose(out_new, out_old, atol=1e-6, rtol=1e-6)


def test_graph_multihead_attention_multicol_matches_expanded_reference():
    torch.manual_seed(0)
    B, T, C, D, H = 2, 6, 3, 16, 4

    model = GraphMultiheadAttention(d_model=D, nhead=H, dropout=0.0, learnable_residual=True)
    model.eval()

    src = torch.randn(B, T, C, D)
    edge_index_batch = [
        torch.tensor([[0, 1, 2, 2], [1, 2, 0, 3]], dtype=torch.long),
        torch.tensor([[1, 3, 4], [0, 2, 5]], dtype=torch.long),
    ]

    with torch.no_grad():
        out_new = model(src, edge_index_batch)
        alpha = torch.sigmoid(model.alpha)

        src_bc = src.permute(0, 2, 1, 3).reshape(B * C, T, D)
        out_ref_bc = torch.empty_like(src_bc)
        for bc in range(B * C):
            b = bc // C
            x = src_bc[bc]
            edge_index = edge_index_batch[b].to(device=x.device, dtype=torch.long)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]

            if edge_src.numel() == 0:
                out_ref_bc[bc] = (1.0 - alpha) * x
                continue

            q = model.q_proj(x).view(T, model.nhead, model.head_dim)
            k = model.k_proj(x).view(T, model.nhead, model.head_dim)
            v = model.v_proj(x).view(T, model.nhead, model.head_dim)

            q_dst = q[edge_dst]
            k_src = k[edge_src]
            v_src = v[edge_src]

            attn_logits = (q_dst * k_src).sum(dim=-1) * model.scale

            edge_weight = torch.zeros_like(attn_logits)
            unique_dst = torch.unique(edge_dst)
            for h in range(model.nhead):
                logits_h = attn_logits[:, h]
                for dst_idx in unique_dst.tolist():
                    mask = edge_dst == dst_idx
                    if mask.any():
                        edge_weight[mask, h] = torch.softmax(logits_h[mask], dim=0)

            messages = v_src * edge_weight.unsqueeze(-1)
            agg = torch.zeros((T, model.nhead, model.head_dim), dtype=src.dtype, device=src.device)
            for h in range(model.nhead):
                agg[:, h, :].index_add_(0, edge_dst, messages[:, h, :])

            attn_out = model.out_proj(agg.reshape(T, model.d_model))
            out_ref_bc[bc] = (1.0 - alpha) * x + alpha * attn_out

        out_ref = out_ref_bc.reshape(B, C, T, D).permute(0, 2, 1, 3)

    assert torch.allclose(out_new, out_ref, atol=1e-6, rtol=1e-6)


def test_graph_multihead_attention_chunked_matches_non_chunked():
    torch.manual_seed(0)
    B, T, C, D, H = 2, 64, 2, 16, 4

    src = torch.randn(B, T, C, D)
    dense_edges = torch.cartesian_prod(torch.arange(T), torch.arange(T)).T
    edge_index_batch = [dense_edges.clone(), dense_edges.clone()]

    model_non_chunked = GraphMultiheadAttention(
        d_model=D, nhead=H, dropout=0.0, max_parallel_edges=10**9, learnable_residual=True
    )
    model_chunked = GraphMultiheadAttention(
        d_model=D, nhead=H, dropout=0.0, max_parallel_edges=2048, learnable_residual=True
    )
    model_chunked.load_state_dict(model_non_chunked.state_dict())
    model_non_chunked.eval()
    model_chunked.eval()

    with torch.no_grad():
        out_non_chunked = model_non_chunked(src, edge_index_batch)
        out_chunked = model_chunked(src, edge_index_batch)

    assert torch.allclose(out_chunked, out_non_chunked, atol=1e-6, rtol=1e-6)


def test_graph_attention_block_alpha_initialization_and_forward_shape():
    torch.manual_seed(0)
    block = GraphAttentionBlock(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        norm_first=True,
        learnable_residual=True,
    )

    alpha = torch.sigmoid(block.attn.alpha.detach().cpu())
    assert torch.isclose(alpha, torch.tensor(0.2), atol=1e-8)

    src = torch.randn(2, 6, 1, 16)
    edge_index_batch = [
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[0, 2, 4], [2, 4, 5]], dtype=torch.long),
    ]

    out = block(src, edge_index_batch)
    assert out.shape == src.shape
    assert torch.isfinite(out).all()


def test_graph_attention_uses_standard_residual_by_default():
    model = GraphMultiheadAttention(d_model=16, nhead=4, dropout=0.0)
    assert not hasattr(model, "alpha")
    assert model.learnable_residual is False

    src = torch.randn(2, 5, 1, 16)
    edge_index_batch = [
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
    ]

    out = model(src, edge_index_batch)
    assert out.shape == src.shape
    assert torch.allclose(out[1], src[1])
    assert torch.isfinite(out).all()


def test_tabicl_graph_backend_forward_with_column_identity_rotation():
    torch.manual_seed(0)

    batch_size = 2
    train_size = 8
    total_rows = 12
    num_features = 6

    X = torch.randn(batch_size, total_rows, num_features)
    y_train = _build_labels(batch_size=batch_size, train_size=train_size, num_classes=2)
    d = torch.tensor([4, 6], dtype=torch.long)

    model = TabICL(
        max_classes=5,
        max_features=num_features,
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=4,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=4,
        row_num_cls=4,
        icl_num_blocks=1,
        icl_nhead=4,
        icl_backend="graph",
    )
    model.train()

    graph_set = build_class_conditioned_graphs(
        y_train=y_train,
        total_nodes=total_rows,
        num_graphs=model.graph_num_graphs,
        min_train_neighbors=model.graph_min_train_neighbors,
        max_train_neighbors=model.graph_max_train_neighbors,
        cross_label_fraction=model.graph_cross_label_fraction,
        train_neighbors_per_test=model.graph_train_neighbors_per_test,
        seed=model.graph_seed,
    )
    out = model(X, y_train, d=d, graph_set=graph_set)
    assert out.shape == (batch_size, total_rows - train_size, 5)
    assert torch.isfinite(out).all()
