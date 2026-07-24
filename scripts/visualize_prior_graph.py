from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required for visualization. Install with `uv pip install matplotlib`.") from exc

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError("networkx is required for graph visualization. Install with `uv pip install networkx`.") from exc

from tabicl.prior._dataset import PriorDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize class-conditioned graph for first dataset in a prior batch.")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of datasets to draw from prior")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Datasets per group")
    parser.add_argument("--min_features", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=100)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--min_train_size", type=float, default=0.1)
    parser.add_argument("--max_train_size", type=float, default=0.9)
    parser.add_argument(
        "--prior_type",
        type=str,
        default="mix_scm",
        choices=["mlp_scm", "tree_scm", "mix_scm", "nanotabicl"],
    )
    parser.add_argument("--prior_device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph_min_train_neighbors", type=int, default=8)
    parser.add_argument("--graph_max_train_neighbors", type=int, default=15)
    parser.add_argument("--graph_same_label_ratio", type=float, default=0.9)
    parser.add_argument("--graph_cross_label_ratio", type=float, default=0.1)
    parser.add_argument("--graph_test_k_per_class", type=int, default=8)
    parser.add_argument("--c", type=int, default=3)
    parser.add_argument(
        "--plot_test_fraction",
        type=float,
        default=0.1,
        help="Fraction of test nodes to display in the plot (0-1). All train nodes are always shown.",
    )
    parser.add_argument("--output", type=str, default="scripts/graph_preview.png")
    return parser


def _to_numpy_color_labels(y_train: torch.Tensor, train_size: int, total_nodes: int) -> np.ndarray:
    labels = np.full(total_nodes, -1, dtype=int)
    labels[:train_size] = y_train[:train_size].cpu().numpy().astype(int)
    return labels


def _select_plot_nodes(seq_len: int, train_size: int, test_fraction: float, seed: int) -> np.ndarray:
    train_nodes = np.arange(train_size)
    test_nodes = np.arange(train_size, seq_len)

    if len(test_nodes) == 0:
        return train_nodes

    test_fraction = min(max(test_fraction, 0.0), 1.0)
    keep_test = int(round(len(test_nodes) * test_fraction))

    if test_fraction > 0.0:
        keep_test = max(1, keep_test)

    if keep_test >= len(test_nodes):
        chosen_test = test_nodes
    elif keep_test == 0:
        chosen_test = np.array([], dtype=int)
    else:
        rng = np.random.default_rng(seed)
        chosen_test = np.sort(rng.choice(test_nodes, size=keep_test, replace=False))

    return np.concatenate([train_nodes, chosen_test])


def _label_group_layout(g_plot: nx.DiGraph, label_colors: np.ndarray, train_size: int, seed: int) -> dict[int, np.ndarray]:
    train_nodes = [n for n in g_plot.nodes if n < train_size]
    test_nodes = [n for n in g_plot.nodes if n >= train_size]

    present_labels = sorted({int(label_colors[n]) for n in train_nodes})

    groups: list[tuple[str, int | None, list[int]]] = []
    for label in present_labels:
        label_nodes = [n for n in train_nodes if int(label_colors[n]) == label]
        if label_nodes:
            groups.append(("label", label, label_nodes))
    if test_nodes:
        groups.append(("test", None, test_nodes))

    if not groups:
        return nx.spring_layout(g_plot, seed=seed)

    n_groups = len(groups)
    radius = max(2.5, 1.2 * n_groups)
    centers: list[np.ndarray] = []
    for i in range(n_groups):
        theta = 2.0 * np.pi * i / n_groups
        centers.append(np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=float))

    pos: dict[int, np.ndarray] = {}
    for i, (_, _, nodes) in enumerate(groups):
        center = centers[i]
        if len(nodes) == 1:
            pos[nodes[0]] = center
            continue

        subgraph = g_plot.subgraph(nodes)
        local_pos = nx.spring_layout(
            subgraph,
            seed=seed + i,
            k=max(0.2, 1.6 / np.sqrt(max(2, len(nodes)))),
            iterations=120,
        )
        spread = max(0.45, 0.16 * np.sqrt(len(nodes)))
        for node, xy in local_pos.items():
            pos[node] = center + spread * np.asarray(xy, dtype=float)

    return pos


def main() -> None:
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = PriorDataset(
        batch_size=args.batch_size,
        batch_size_per_gp=args.batch_size_per_gp,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        max_seq_len=args.max_seq_len,
        min_train_size=args.min_train_size,
        max_train_size=args.max_train_size,
        prior_type=args.prior_type,
        device=args.prior_device,
        n_jobs=1,
        graph_backend=True,
        graph_min_train_neighbors=args.graph_min_train_neighbors,
        graph_max_train_neighbors=args.graph_max_train_neighbors,
        graph_same_label_ratio=args.graph_same_label_ratio,
        graph_cross_label_ratio=args.graph_cross_label_ratio,
        graph_test_k_per_class=args.graph_test_k_per_class,
        graph_seed=args.seed,
    )

    _, y, _, seq_lens, train_sizes, graph_sets = dataset.get_batch(batch_size=args.batch_size)

    seq_len = int(seq_lens[0].item())
    train_size = int(train_sizes[0].item())
    y0 = y[0, :seq_len].long()
    y_train = y0[:train_size].unsqueeze(0)

    # Use the graph generated together with the sampled prior dataset so the
    # visualization reflects the exact graph configuration and random draw.
    graph = graph_sets[0]

    edge_index = graph.graphs[0].edge_index[0].cpu()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    g = nx.DiGraph()
    g.add_nodes_from(range(seq_len))
    g.add_edges_from(zip(src, dst))

    plot_nodes = _select_plot_nodes(seq_len=seq_len, train_size=train_size, test_fraction=args.plot_test_fraction, seed=args.seed)
    g_plot = g.subgraph(plot_nodes.tolist()).copy()

    label_colors = _to_numpy_color_labels(y0, train_size, seq_len)

    # Train nodes by label colormap, test nodes in grey
    cmap = plt.get_cmap("tab20")

    node_colors = []
    for idx in g_plot.nodes:
        if idx >= train_size:
            node_colors.append("#808080")
        else:
            node_colors.append(cmap((label_colors[idx] % 20) / 20.0))

    plot_node_count = g_plot.number_of_nodes()
    pos = _label_group_layout(g_plot=g_plot, label_colors=label_colors, train_size=train_size, seed=args.seed)

    edge_colors = []
    for source, target in g_plot.edges:
        same_class = (
            source < train_size
            and target < train_size
            and label_colors[source] == label_colors[target]
        )
        edge_colors.append("#D62728" if same_class else "#3A3A3A")

    plt.figure(figsize=(16, 12), dpi=180)
    nx.draw_networkx_edges(g_plot, pos, alpha=0.45, width=1.3, edge_color=edge_colors, arrows=False)
    nx.draw_networkx_nodes(g_plot, pos, node_color=node_colors, node_size=220)

    # Label a subset of nodes for readability
    sample_label_count = min(plot_node_count, 60)
    labeled_nodes = list(g_plot.nodes)[:sample_label_count]
    node_labels = {i: str(i) for i in labeled_nodes}
    nx.draw_networkx_labels(g_plot, pos, labels=node_labels, font_size=8, font_color="black")

    train_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10)
    test_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#808080", markersize=10)
    plt.legend([train_patch, test_patch], ["Train nodes (class-colored)", "Test nodes (grey)"], loc="upper right")

    plt.title(
        f"Class-Conditioned Graph (first dataset)\n"
        f"nodes_shown={plot_node_count} (train={train_size}, test={plot_node_count - train_size}), "
        f"edges_shown={g_plot.number_of_edges()}, total_nodes={seq_len}, total_edges={edge_index.shape[1]}"
    )
    plt.axis("off")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved graph visualization to: {out_path}")


if __name__ == "__main__":
    main()
