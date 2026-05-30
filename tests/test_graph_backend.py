import torch
import pytest

from src.tabicl._model.graph import build_class_conditioned_graph
from src.tabicl._model.learning import ICLearning


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

    # Train node incoming train-neighbor count should stay in [8, 15] excluding self loops.
    for train_node in range(train_size):
        mask = (dst == train_node) & (src < train_size) & (src != train_node)
        degree = int(mask.sum().item())
        assert 8 <= degree <= 15

    # Each test node should connect to at least 3 train nodes from each class.
    labels = y_train[0]
    classes = torch.unique(labels)
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
    R = torch.randn(batch_size, total_nodes, d_model)

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

    out = model(R, y_train)
    assert out.shape == (batch_size, total_nodes - train_size, 5)
    assert torch.isfinite(out).all()


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
