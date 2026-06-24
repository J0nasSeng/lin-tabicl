import torch
import pytest

from src.tabicl._model.graph import build_class_conditioned_graph
from src.tabicl._model.gat import GraphMultiheadAttention
from src.tabicl._model.gat import GraphAttentionBlock
from src.tabicl._model.learning import ICLearning
from src.tabicl._model.tabicl import TabICL
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
        same_label_ratio=0.9,
        cross_label_ratio=0.1,
        test_k_per_class=3,
        seed=7,
    )

    edge_index = graph.edge_index[0]
    src = edge_index[0]
    dst = edge_index[1]

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
        graph_same_label_ratio=0.9,
        graph_cross_label_ratio=0.1,
        graph_test_k_per_class=3,
        graph_seed=0,
    )
    model.train()

    col_embeddings = torch.randn(batch_size, total_nodes, model.graph_num_cls, model.graph_col_dim)
    graph_input = model.prepare_graph_input(col_embeddings=col_embeddings, y_train=y_train)
    out = model(graph_input, y_train)
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


def test_graph_builder_shared_graph_when_labels_identical():
    batch_size = 3
    train_size = 15
    total_nodes = 22
    y_one = _build_labels(batch_size=1, train_size=train_size, num_classes=3)
    y_train = y_one.expand(batch_size, -1).clone()

    graph = build_class_conditioned_graph(
        y_train=y_train,
        total_nodes=total_nodes,
        seed=17,
        share_graph_across_batch=True,
        share_graph_require_identical_labels=True,
    )

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

    graph = build_class_conditioned_graph(
        y_train=y_train,
        total_nodes=total_nodes,
        seed=3,
        share_graph_across_batch=True,
        share_graph_require_identical_labels=True,
    )

    assert len(graph.edge_index) == y_train.shape[0]
    assert not torch.equal(graph.edge_index[0], graph.edge_index[1])


def test_graph_multihead_attention_vectorized_matches_legacy_loop():
    torch.manual_seed(0)
    B, T, C, D, H = 3, 7, 1, 16, 4

    model = GraphMultiheadAttention(d_model=D, nhead=H, dropout=0.0)
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

    model = GraphMultiheadAttention(d_model=D, nhead=H, dropout=0.0)
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


def test_graph_attention_block_alpha_initialization_and_forward_shape():
    torch.manual_seed(0)
    block = GraphAttentionBlock(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        norm_first=True,
    )

    alpha = torch.sigmoid(block.attn.alpha.detach().cpu())
    assert torch.isclose(alpha, torch.tensor(0.05), atol=1e-8)

    src = torch.randn(2, 6, 1, 16)
    edge_index_batch = [
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[0, 2, 4], [2, 4, 5]], dtype=torch.long),
    ]

    out = block(src, edge_index_batch)
    assert out.shape == src.shape
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

    out = model(X, y_train, d=d)
    assert out.shape == (batch_size, total_rows - train_size, 5)
    assert torch.isfinite(out).all()
