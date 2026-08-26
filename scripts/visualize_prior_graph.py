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

from tabicl._model.graph import GraphTopologyPrior
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
    parser.add_argument("--graph_cross_label_fraction", type=float, default=0.1)
    parser.add_argument("--graph_train_neighbors_per_test", type=int, default=8)
    parser.add_argument(
        "--plot_test_fraction",
        type=float,
        default=0.1,
        help="Fraction of test nodes to display in the plot (0-1). All train nodes are always shown.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print graph statistics; do not generate or save a plot.",
    )
    parser.add_argument(
        "--plot-statistics",
        action="store_true",
        help="Generate a 3-by-statistic histogram plot over all sampled datasets.",
    )
    parser.add_argument("--output", type=str, default="scripts/graph_preview.png")
    parser.add_argument(
        "--statistics-output",
        type=str,
        default="scripts/graph_statistics.png",
        help="Output path for the batch statistics histogram plot.",
    )
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


def _log_graph_metrics(
    g: nx.DiGraph,
    labels: np.ndarray,
    train_size: int,
    cross_label_fraction: float,
) -> None:
    """Log degree summaries and label-conditioned edge counts for the full graph."""
    statistics = _graph_statistics(g, labels, train_size)
    print("Graph metrics (full generated graph):")
    print(
        f"  train degree: in={statistics['train_in_degree']:.2f}, "
        f"out={statistics['train_out_degree']:.2f}"
    )
    print(
        f"  test degree:  in={statistics['test_in_degree']:.2f}, "
        f"out={statistics['test_out_degree']:.2f}"
    )
    print(
        f"  all sample edges: intra-label={statistics['all_intra_edges']:.0f}, "
        f"cross-label={statistics['all_cross_edges']:.0f}"
    )
    print(
        f"  train-train sample edges: intra-label={statistics['train_train_intra_edges']:.0f}, "
        f"cross-label={statistics['train_train_cross_edges']:.0f}"
    )
    print(
        f"  train-train cross-label fraction: requested={cross_label_fraction:.3f}, "
        f"realized={statistics['train_train_cross_fraction']:.3f}"
    )
    print(f"  adjusted homophily/assortativity: {statistics['assortativity']:.3f}")


def _graph_statistics(g: nx.DiGraph, labels: np.ndarray, train_size: int) -> dict[str, float]:
    """Return the statistics plotted for one generated graph."""
    train_nodes = list(range(train_size))
    test_nodes = list(range(train_size, len(labels)))

    def _average_degree(nodes: list[int], degree: str) -> float:
        if not nodes:
            return 0.0
        values = g.in_degree(nodes) if degree == "in" else g.out_degree(nodes)
        return float(np.mean([value for _, value in values]))

    train_in_degree = _average_degree(train_nodes, "in")
    train_out_degree = _average_degree(train_nodes, "out")
    test_in_degree = _average_degree(test_nodes, "in")
    test_out_degree = _average_degree(test_nodes, "out")
    intra_label_edges = 0
    cross_label_edges = 0
    train_train_intra_edges = 0
    train_train_cross_edges = 0
    for source, target in g.edges:
        same_label = labels[source] == labels[target]
        if same_label:
            intra_label_edges += 1
        else:
            cross_label_edges += 1

        if source < train_size and target < train_size:
            if same_label:
                train_train_intra_edges += 1
            else:
                train_train_cross_edges += 1

    train_train_total = train_train_intra_edges + train_train_cross_edges
    realized_cross_fraction = (
        train_train_cross_edges / train_train_total if train_train_total else 0.0
    )
    nx.set_node_attributes(g, {node: int(labels[node]) for node in g.nodes}, "label")
    assortativity = nx.attribute_assortativity_coefficient(g, "label")
    if not np.isfinite(assortativity):
        assortativity = 0.0
    return {
        "train_in_degree": train_in_degree,
        "train_out_degree": train_out_degree,
        "test_in_degree": test_in_degree,
        "test_out_degree": test_out_degree,
        "all_intra_edges": float(intra_label_edges),
        "all_cross_edges": float(cross_label_edges),
        "train_train_intra_edges": float(train_train_intra_edges),
        "train_train_cross_edges": float(train_train_cross_edges),
        "train_train_cross_fraction": realized_cross_fraction,
        "assortativity": float(assortativity),
    }


STATISTIC_LABELS = {
    "train_in_degree": "Train mean in-degree",
    "train_out_degree": "Train mean out-degree",
    "test_in_degree": "Test mean in-degree",
    "test_out_degree": "Test mean out-degree",
    "all_intra_edges": "All intra-label edges",
    "all_cross_edges": "All cross-label edges",
    "train_train_intra_edges": "Train-train intra-label edges",
    "train_train_cross_edges": "Train-train cross-label edges",
    "train_train_cross_fraction": "Train-train cross-label fraction",
    "assortativity": "Adjusted homophily / assortativity",
}


def _plot_statistics(
    statistics: dict[str, list[dict[str, float]]],
    output: str,
) -> None:
    """Plot one histogram row per graph mode and one column per statistic."""
    modes = ["tabular v1", "tabular v2", "graph mode"]
    stat_names = list(STATISTIC_LABELS)
    figure, axes = plt.subplots(
        len(modes), len(stat_names), figsize=(4.0 * len(stat_names), 10), squeeze=False, dpi=160
    )
    for row, mode in enumerate(modes):
        values_by_stat = {
            name: np.asarray([item[name] for item in statistics[mode]], dtype=float)
            for name in stat_names
        }
        for column, name in enumerate(stat_names):
            axis = axes[row, column]
            values = values_by_stat[name]
            axis.hist(values, bins=min(15, max(3, len(values))), color=("#4C78A8", "#F58518", "#54A24B")[row], alpha=0.8)
            mean = float(np.mean(values))
            median = float(np.median(values))
            axis.axvline(mean, color="black", linestyle="--", linewidth=1.2, label=f"mean {mean:.2f}")
            axis.axvline(median, color="black", linestyle=":", linewidth=1.5, label=f"median {median:.2f}")
            axis.set_title(STATISTIC_LABELS[name], fontsize=9)
            axis.tick_params(axis="both", labelsize=8)
            axis.legend(fontsize=7, loc="best")
            if column == 0:
                axis.set_ylabel(mode, fontsize=11)
    figure.suptitle("GraphTopologyPrior statistics across sampled datasets", fontsize=16)
    figure.tight_layout()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved statistics histogram plot to: {out_path}")


def _to_networkx(edge_index: torch.Tensor, total_nodes: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(total_nodes))
    graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    return graph


def _draw_graph(
    ax: plt.Axes,
    graph: nx.DiGraph,
    labels: np.ndarray,
    train_size: int,
    test_fraction: float,
    seed: int,
    title: str,
) -> None:
    plot_nodes = _select_plot_nodes(
        seq_len=len(labels), train_size=train_size, test_fraction=test_fraction, seed=seed
    )
    g_plot = graph.subgraph(plot_nodes.tolist()).copy()
    cmap = plt.get_cmap("tab20")
    node_colors = [
        "#808080" if node >= train_size else cmap((labels[node] % 20) / 20.0)
        for node in g_plot.nodes
    ]
    pos = _label_group_layout(g_plot=g_plot, label_colors=labels, train_size=train_size, seed=seed)
    edge_colors = [
        "#D62728"
        if source < train_size
        and target < train_size
        and labels[source] == labels[target]
        else "#3A3A3A"
        for source, target in g_plot.edges
    ]
    nx.draw_networkx_edges(
        g_plot, pos, ax=ax, alpha=0.45, width=1.0, edge_color=edge_colors, arrows=False
    )
    nx.draw_networkx_nodes(g_plot, pos, ax=ax, node_color=node_colors, node_size=100)
    labeled_nodes = list(g_plot.nodes)[: min(g_plot.number_of_nodes(), 40)]
    nx.draw_networkx_labels(
        g_plot,
        pos,
        ax=ax,
        labels={node: str(node) for node in labeled_nodes},
        font_size=6,
        font_color="black",
    )
    ax.set_title(
        f"{title}\nshown={g_plot.number_of_nodes()}, edges={g_plot.number_of_edges()}"
    )
    ax.axis("off")


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
        graph_backend=False,
        graph_min_train_neighbors=args.graph_min_train_neighbors,
        graph_max_train_neighbors=args.graph_max_train_neighbors,
        graph_cross_label_fraction=args.graph_cross_label_fraction,
        graph_train_neighbors_per_test=args.graph_train_neighbors_per_test,
        graph_seed=args.seed,
    )

    _, y, _, seq_lens, train_sizes = dataset.get_batch(batch_size=args.batch_size)

    seq_len = int(seq_lens[0].item())
    train_size = int(train_sizes[0].item())
    y0 = y[0, :seq_len].long()

    label_colors = _to_numpy_color_labels(y0, train_size, seq_len)
    labels = y[0, :seq_len].long().unsqueeze(0)

    # Keep all three samples based on the same generated dataset. This makes
    # differences in topology attributable to GraphTopologyPrior rather than the SCM.
    priors = {
        "tabular v1": GraphTopologyPrior(
            graph_v1_prob=1.0,
            graph_v2_prob=0.0,
            graph_prob=0.0,
            min_train_neighbors=args.graph_min_train_neighbors,
            max_train_neighbors=args.graph_max_train_neighbors,
            cross_label_fraction=args.graph_cross_label_fraction,
            train_neighbors_per_test=args.graph_train_neighbors_per_test,
            seed=args.seed,
        ),
        "tabular v2": GraphTopologyPrior(
            graph_v1_prob=0.0,
            graph_v2_prob=1.0,
            graph_prob=0.0,
            min_train_neighbors=args.graph_min_train_neighbors,
            max_train_neighbors=args.graph_max_train_neighbors,
            cross_label_fraction=args.graph_cross_label_fraction,
            train_neighbors_per_test=args.graph_train_neighbors_per_test,
            seed=args.seed + 1,
        ),
        "graph mode": GraphTopologyPrior(
            graph_v1_prob=0.0,
            graph_v2_prob=0.0,
            graph_prob=1.0,
            min_train_neighbors=args.graph_min_train_neighbors,
            max_train_neighbors=args.graph_max_train_neighbors,
            cross_label_fraction=args.graph_cross_label_fraction,
            train_neighbors_per_test=args.graph_train_neighbors_per_test,
            seed=args.seed + 2,
        ),
    }

    if args.plot_statistics:
        # Generate one graph per mode for every dataset in the sampled batch.
        # The prior dataset normally uses a shared sequence length and split
        # within a batch; retain a clear error if a custom prior violates that
        # assumption, since the compact graph payload requires rectangular
        # labels.
        batch_seq_lens = seq_lens[: args.batch_size].long().tolist()
        batch_train_sizes = train_sizes[: args.batch_size].long().tolist()
        if len(set(batch_seq_lens)) != 1 or len(set(batch_train_sizes)) != 1:
            raise ValueError("--plot-statistics requires one sequence length and train size per batch")
        batch_labels = y[:, :seq_len].long()
        batch_statistics: dict[str, list[dict[str, float]]] = {}
        for mode, prior in priors.items():
            graph_set = prior(batch_labels, train_size, num_graphs=1)
            mode_statistics = []
            for dataset_index in range(batch_labels.shape[0]):
                edge_index = graph_set.graphs[0].edge_index[dataset_index].cpu()
                graph = _to_networkx(edge_index, seq_len)
                dataset_labels = batch_labels[dataset_index].cpu().numpy().astype(int)
                mode_statistics.append(_graph_statistics(graph, dataset_labels, train_size))
            batch_statistics[mode] = mode_statistics

            values = {
                name: float(np.mean([item[name] for item in mode_statistics]))
                for name in STATISTIC_LABELS
            }
            print(f"\n{mode} batch means:")
            for name, value in values.items():
                print(f"  {STATISTIC_LABELS[name]}: {value:.3f}")

        if not args.no_plot:
            _plot_statistics(batch_statistics, args.statistics_output)
        return

    graphs = {
        name: prior(labels, train_size, num_graphs=1).graphs[0].edge_index[0].cpu()
        for name, prior in priors.items()
    }
    networkx_graphs = {
        name: _to_networkx(edge_index, seq_len) for name, edge_index in graphs.items()
    }

    for name, graph in networkx_graphs.items():
        print(f"\n{name}:")
        _log_graph_metrics(
            g=graph,
            labels=label_colors,
            train_size=train_size,
            cross_label_fraction=args.graph_cross_label_fraction,
        )

    if args.no_plot:
        return

    figure, axes = plt.subplots(1, 3, figsize=(24, 9), dpi=180)
    for axis, (name, graph) in zip(axes, networkx_graphs.items()):
        _draw_graph(
            ax=axis,
            graph=graph,
            labels=label_colors,
            train_size=train_size,
            test_fraction=args.plot_test_fraction,
            seed=args.seed,
            title=name,
        )
    figure.suptitle(
        f"GraphTopologyPrior comparison (total_nodes={seq_len}, train_nodes={train_size})",
        fontsize=16,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)

    print(f"Saved graph visualization to: {out_path}")


if __name__ == "__main__":
    main()
