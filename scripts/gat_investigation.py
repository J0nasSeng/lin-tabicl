"""Isolated investigation of graph-1d GAT layers in a frozen TabICL checkpoint.

The script evaluates one prior-sampled dataset with skip-GAT, cumulative GAT
prefixes, and the full GAT stack. Prefixes are applied externally by replacing
``graph_blocks`` temporarily; no prefix-specific model code is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, recall_score

from tabicl import TabICLClassifier
from tabicl.prior._dataset import PriorDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--encoder-checkpoint", type=Path, default=None,
        help="Optional custom encoder-backend TabICL checkpoint to evaluate on the same datasets.",
    )
    parser.add_argument("--num-datasets", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-size-per-gp", type=int, default=1)
    parser.add_argument("--min-features", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=256)
    parser.add_argument("--min-seq-len", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--min-train-size", type=float, default=0.1)
    parser.add_argument("--max-train-size", type=float, default=0.9)
    parser.add_argument("--prior-type", default="nanotabicl")
    parser.add_argument("--prior-device", default="cpu")
    parser.add_argument("--normalization", choices=("none", "std", "robust"), default="std")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument(
        "--num-graphs", type=int, default=None,
        help="Number of graph slots used throughout the GAT stack. Must divide the block count.",
    )
    parser.add_argument(
        "--show-graph", action="store_true",
        help="Overlay graph edges on UMAP plots. Requires --num-graphs 1.",
    )
    parser.add_argument(
        "--graph-ablation", action="store_true",
        help="Run empty, fully-connected, 70%% cross-label, and 10%% cross-label graph ablations.",
    )
    parser.add_argument("--graph-ablation-temperature", type=float, default=1.0)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        help="Attention temperatures to ablate. Values below 1 sharpen attention.",
    )
    parser.add_argument(
        "--attention-top-k", type=int, nargs="+", default=[1, 3, 5],
        help="Top-k attention sizes for same-label attention accuracy statistics.",
    )
    parser.add_argument(
        "--blocks", type=int, nargs="+", default=None,
        help="GAT prefix lengths. 0 means skip-GAT; default is 0 and every prefix.",
    )
    parser.add_argument("--umap-n-neighbors", type=int, default=10)
    parser.add_argument("--umap-n-epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("gat_investigation"))
    return parser


def _regular_batch(batch: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(
        value.to_padded_tensor(padding=0.0)
        if isinstance(value, torch.Tensor) and value.is_nested else value
        for value in batch
    )


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    return float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _build_classifier(
    checkpoint: Path,
    device: str,
    seed: int,
    skip_gat: bool = False,
    graph_config: dict[str, object] | None = None,
) -> TabICLClassifier:
    config = dict(graph_config or {})
    if skip_gat:
        config["skip_gat"] = True
    return TabICLClassifier(
        model_path=checkpoint,
        n_estimators=1,
        batch_size=1,
        device=device,
        random_state=seed,
        n_jobs=1,
        norm_methods="none",
        gat_mode="ensemble",
        gat_num_iterations=1,
        gat_entry_layer=None,
        graph_config=config or None,
    )

def _build_encoder_classifier(
    checkpoint: Path, device: str, seed: int, n_estimators: int
) -> TabICLClassifier:
    """Build the optional custom encoder-backend comparison model."""
    return TabICLClassifier(
        model_path=checkpoint,
        n_estimators=n_estimators,
        batch_size=1,
        device=device,
        random_state=seed,
        n_jobs=1,
        norm_methods="none",
    )


def _encoder_layer_representations(
    classifier: TabICLClassifier, X: np.ndarray
) -> dict[str, np.ndarray]:
    """Return test-row representations after each encoder ICL block."""
    model = classifier.model_
    blocks = model.icl_predictor.tf_icl.blocks
    captured: list[list[torch.Tensor]] = [[] for _ in blocks]

    hooks = []
    for index, block in enumerate(blocks):
        hooks.append(
            block.register_forward_hook(
                lambda _module, _inputs, output, index=index: captured[index].append(
                    output.detach() if isinstance(output, torch.Tensor) else output[0].detach()
                )
            )
        )
    try:
        X = classifier.X_encoder_.transform(X)
        if classifier.feature_reducer_ is not None:
            X = classifier.feature_reducer_.transform(X)
        for subset, generator in zip(classifier.feature_subsets_, classifier.ensemble_generators_):
            data = generator.transform(X[:, subset], mode="both")
            for norm_method, (Xs, ys) in data.items():
                feature_shuffles = generator.feature_shuffles_[norm_method]
                batch_size = classifier.batch_size or Xs.shape[0]
                n_batches = int(np.ceil(Xs.shape[0] / batch_size))
                X_batches = np.array_split(Xs, n_batches)
                y_batches = np.array_split(ys, n_batches)
                shuffle_batches = (
                    np.array_split(feature_shuffles, n_batches)
                    if feature_shuffles is not None
                    else [None] * n_batches
                )
                for X_batch, y_batch, shuffle_batch in zip(
                    X_batches, y_batches, shuffle_batches,
                ):
                    X_batch = torch.from_numpy(X_batch).float().to(classifier.device_)
                    y_batch = torch.from_numpy(y_batch).float().to(classifier.device_)
                    if shuffle_batch is not None:
                        shuffle_batch = shuffle_batch.tolist()
                    with torch.no_grad():
                        model(
                            X=X_batch,
                            y_train=y_batch,
                            feature_shuffles=shuffle_batch,
                            return_logits=True,
                            inference_config=classifier.inference_config_,
                            return_pre_decoder_repr=True,
                        )
    finally:
        for hook in hooks:
            hook.remove()

    return {
        f"encoder_{index + 1}": np.mean(
            np.concatenate([value.float().cpu().numpy() for value in values], axis=0), axis=0
        )
        for index, values in enumerate(captured)
        if values
    }


def _prefixes(classifier: TabICLClassifier, requested: list[int] | None) -> list[int]:
    gat = classifier.model_.model.icl_predictor.gat_icl
    count = len(gat.graph_blocks)
    values = requested if requested is not None else [0, *range(1, count + 1)]
    values = list(dict.fromkeys(values))
    if any(value < 0 or value > count for value in values):
        raise ValueError(f"--blocks must be between 0 and {count}")
    return values


def _set_temperature(classifier: TabICLClassifier, temperature: float) -> None:
    """Set the inference temperature on every GAT attention module."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Attention temperatures must be positive, got {temperature}")
    gat = classifier.model_.model.icl_predictor.gat_icl
    for block in gat.graph_blocks:
        block.attn.temperature = temperature


def _run_prefix(
    classifier: TabICLClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    prefix: int,
    labels: np.ndarray,
    temperature: float,
    attention_top_k: list[int],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run standard inference with a temporary graph-block prefix."""
    engine = classifier.model_
    gat = engine.model.icl_predictor.gat_icl
    original_blocks = gat.graph_blocks
    try:
        if prefix == 0:
            raise ValueError("prefix zero must use the separately constructed skip-GAT classifier")
        gat.graph_blocks = torch.nn.ModuleList(list(original_blocks)[:prefix])
        classifier.fit(x_train, y_train)
        _set_temperature(classifier, temperature)
        probabilities = classifier.predict_proba(x_test)
        representation = classifier.predict_representation(x_test)
        attention = _attention_statistics(
            engine, labels, y_train.shape[0], attention_top_k
        )
        return probabilities, representation, attention
    finally:
        gat.graph_blocks = original_blocks


def _cached_graph_edges(classifier: TabICLClassifier) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the graph edges cached by the last GAT attention invocation."""
    gat = classifier.model_.model.icl_predictor.gat_icl
    for block in reversed(gat.graph_blocks):
        attention = block.attn
        src = getattr(attention, "last_attention_edge_src", None)
        dst = getattr(attention, "last_attention_edge_dst", None)
        if src is not None and dst is not None:
            return src.detach().cpu().numpy(), dst.detach().cpu().numpy()
    return None


def _attention_statistics(
    engine: object, labels: np.ndarray, train_size: int, attention_top_k: list[int]
) -> list[dict]:
    """Aggregate attention and compare it with the incoming graph composition.

    The uniform baseline is computed independently for every destination from
    its actual incoming edges.  This is intentionally not based on
    ``num_classes``: a task may contain fewer active classes, and graph modes
    can produce different candidate sets.
    """
    predictor = engine.model.icl_predictor
    gat = predictor.gat_icl
    y_all = np.asarray(labels)
    # The cache is stored on each GraphAttentionBlock's attention module. The
    # graph-1d path has one column, while the cache retains a column dimension.
    records: list[dict] = []
    top_k_values = sorted(set(attention_top_k))
    if any(k <= 0 for k in top_k_values):
        raise ValueError(f"Top-k attention sizes must be positive, got {attention_top_k}")
    for layer_index, block in enumerate(gat.graph_blocks):
        attention = block.attn
        weights = getattr(attention, "last_attention_weights", None)
        src = getattr(attention, "last_attention_edge_src", None)
        dst = getattr(attention, "last_attention_edge_dst", None)
        if weights is None or src is None or dst is None:
            continue
        weights_np = weights[:, 0].mean(axis=1).cpu().numpy()
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        same_train: list[float] = []
        different_train: list[float] = []
        true_test: list[float] = []
        wrong_test: list[float] = []
        uniform_train: list[float] = []
        uniform_test: list[float] = []
        train_enrichment: list[float] = []
        test_enrichment: list[float] = []
        incoming_edges_train: list[int] = []
        incoming_edges_test: list[int] = []
        entropy_train: list[float] = []
        entropy_test: list[float] = []
        top_k_train = {k: [] for k in top_k_values}
        top_k_test = {k: [] for k in top_k_values}
        for destination in np.unique(dst_np):
            mask = dst_np == destination
            source = src_np[mask]
            mass = weights_np[mask]
            if destination >= y_all.size:
                continue
            valid = source < y_all.size
            source, mass = source[valid], mass[valid]
            if mass.size == 0:
                continue
            same_mask = y_all[source] == y_all[destination]
            uniform_same = float(np.mean(same_mask))
            observed_same = float(mass[same_mask].sum())
            probabilities = mass / max(float(mass.sum()), 1e-12)
            entropy = float(-np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-12, None))
            ))
            ranked = np.argsort(-probabilities)
            top_k_accuracy = {
                k: float(np.mean(same_mask[ranked[:min(k, mass.size)]]))
                for k in top_k_values
            }
            if destination < train_size:
                same_train.append(observed_same)
                different_train.append(float(mass.sum() - observed_same))
                uniform_train.append(uniform_same)
                train_enrichment.append(observed_same / max(uniform_same, 1e-12))
                incoming_edges_train.append(int(mass.size))
                entropy_train.append(entropy)
                for k, accuracy in top_k_accuracy.items():
                    top_k_train[k].append(accuracy)
            else:
                true_test.append(observed_same)
                wrong_test.append(float(mass.sum() - observed_same))
                uniform_test.append(uniform_same)
                test_enrichment.append(observed_same / max(uniform_same, 1e-12))
                incoming_edges_test.append(int(mass.size))
                entropy_test.append(entropy)
                for k, accuracy in top_k_accuracy.items():
                    top_k_test[k].append(accuracy)
        record = {
            "layer": layer_index,
            "train_destinations": len(same_train),
            "train_same_label_attention": float(np.mean(same_train)) if same_train else None,
            "train_different_label_attention": float(np.mean(different_train)) if different_train else None,
            "train_uniform_same_label_attention": float(np.mean(uniform_train)) if uniform_train else None,
            "train_attention_enrichment": float(np.mean(train_enrichment)) if train_enrichment else None,
            "train_mean_incoming_edges": float(np.mean(incoming_edges_train)) if incoming_edges_train else None,
            "train_attention_entropy": float(np.mean(entropy_train)) if entropy_train else None,
            "test_destinations": len(true_test),
            "test_true_label_attention": float(np.mean(true_test)) if true_test else None,
            "test_wrong_label_attention": float(np.mean(wrong_test)) if wrong_test else None,
            "test_uniform_true_label_attention": float(np.mean(uniform_test)) if uniform_test else None,
            "test_attention_enrichment": float(np.mean(test_enrichment)) if test_enrichment else None,
            "test_mean_incoming_edges": float(np.mean(incoming_edges_test)) if incoming_edges_test else None,
            "test_attention_entropy": float(np.mean(entropy_test)) if entropy_test else None,
        }
        for k in top_k_values:
            record[f"train_top_{k}_same_label_accuracy"] = (
                float(np.mean(top_k_train[k])) if top_k_train[k] else None
            )
            record[f"test_top_{k}_same_label_accuracy"] = (
                float(np.mean(top_k_test[k])) if top_k_test[k] else None
            )
        records.append(record)
    return records


def _plot_umap(
    representations: dict[str, np.ndarray], labels: np.ndarray, train_size: int,
    output: Path, seed: int, n_neighbors: int, n_epochs: int,
    graph_edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from umap import UMAP

    variants = list(representations)
    fig, axes = plt.subplots(2, len(variants), figsize=(5 * len(variants), 8), squeeze=False, dpi=150)
    for column, variant in enumerate(variants):
        coords = UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(labels) - 1),
            n_epochs=n_epochs,
            metric="euclidean",
            n_jobs=1,
            random_state=seed,
        ).fit_transform(np.nan_to_num(representations[variant]))
        for row, (start, end, title) in enumerate(((0, train_size, "Train"), (train_size, len(labels), "Test"))):
            if graph_edges is not None:
                edge_src, edge_dst = graph_edges
                valid_edges = (
                    (edge_src >= start) & (edge_src < end)
                    & (edge_dst >= start) & (edge_dst < end)
                )
                for source, destination in zip(
                    edge_src[valid_edges], edge_dst[valid_edges]
                ):
                    axes[row, column].plot(
                        (coords[source, 0], coords[destination, 0]),
                        (coords[source, 1], coords[destination, 1]),
                        color="black", alpha=0.12, linewidth=0.5, zorder=1,
                    )
            axes[row, column].scatter(
                coords[start:end, 0], coords[start:end, 1], c=labels[start:end],
                cmap="tab10", s=16, alpha=0.8, zorder=2,
            )
            axes[row, column].set(title=f"{variant}: {title}", xlabel="UMAP-1", ylabel="UMAP-2")
            axes[row, column].grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_metrics(rows: list[dict], output_dir: Path) -> None:
    """Plot prediction metrics, with one line for each temperature."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    temperatures = sorted({row["temperature"] for row in rows if row["temperature"] is not None})
    block_values = sorted({row["blocks"] for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
    for axis, metric, title, ylabel in zip(
        axes,
        ("cross_entropy", "balanced_accuracy"),
        ("Prediction cross-entropy", "Prediction balanced accuracy"),
        ("Cross-entropy (lower is better)", "Balanced accuracy (higher is better)"),
    ):
        skip_values = np.asarray(
            [row[metric] for row in rows if row["blocks"] == 0], dtype=float
        )
        if skip_values.size:
            axis.errorbar(
                [0], [np.nanmean(skip_values)], yerr=[np.nanstd(skip_values)],
                marker="o", capsize=4, linewidth=2, color="black", label="skip-GAT",
            )
        encoder_values = np.asarray(
            [row[metric] for row in rows if row["variant"] == "encoder"], dtype=float
        )
        if encoder_values.size:
            axis.errorbar(
                [-1], [np.nanmean(encoder_values)], yerr=[np.nanstd(encoder_values)],
                marker="o", capsize=4, linewidth=2, color="#70AD47",
                label="custom encoder",
            )
        for temperature in temperatures:
            means = []
            stds = []
            plotted_blocks = []
            for block in block_values:
                if block == 0:
                    continue
                values = np.asarray(
                    [row[metric] for row in rows
                     if row["blocks"] == block and row["temperature"] == temperature],
                    dtype=float,
                )
                if values.size == 0:
                    continue
                plotted_blocks.append(block)
                means.append(np.nanmean(values))
                stds.append(np.nanstd(values))
            if means:
                axis.errorbar(
                    plotted_blocks, means, yerr=stds, marker="o", capsize=4,
                    linewidth=2, label=f"temperature={temperature:g}",
                )
        axis.set_xticks(
            [-1, *block_values],
            ["custom encoder", *[
                "skip-GAT" if block == 0 else str(block) for block in block_values
            ]],
        )
        axis.set_xlabel("Number of applied GAT blocks")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.suptitle("GAT depth prediction metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "metrics_vs_gat_blocks.png")
    plt.close(fig)


def _plot_attention_statistics(records: list[dict], output_dir: Path) -> None:
    """Plot label-consistent attention mass by depth and temperature."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not records:
        return
    temperatures = sorted({record["temperature"] for record in records})
    blocks = sorted({int(record["blocks"]) for record in records})
    series = (
        ("train_same_label_attention", "Train: same label", "#4472C4"),
        ("test_true_label_attention", "Test: true label", "#ED7D31"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, constrained_layout=True)
    for axis, split, title in zip(
        axes,
        ("train", "test"),
        ("Training destinations", "Test destinations"),
    ):
        for field, label, color in series:
            if split == "train" and not field.startswith("train_"):
                continue
            if split == "test" and not field.startswith("test_"):
                continue
            for temperature in temperatures:
                means = []
                stds = []
                plotted_blocks = []
                for block in blocks:
                    values = np.asarray(
                        [row[field] for row in records
                         if row["blocks"] == block
                         and row["temperature"] == temperature
                         and row.get(field) is not None],
                        dtype=float,
                    )
                    if values.size == 0:
                        continue
                    plotted_blocks.append(block)
                    means.append(np.nanmean(values))
                    stds.append(np.nanstd(values))
                if means:
                    axis.errorbar(
                        plotted_blocks, means, yerr=stds, marker="o", capsize=4,
                        linewidth=2, color=color, label=f"{label}, T={temperature:g}",
                    )
        axis.set_xlabel("Number of applied GAT blocks")
        axis.set_ylabel("Mean incoming attention mass")
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Label-consistent GAT attention")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "attention_vs_gat_blocks.png")
    plt.close(fig)


def _plot_attention_enrichment(records: list[dict], output_dir: Path) -> None:
    """Plot attention enrichment by GAT depth and temperature."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not records:
        return
    temperatures = sorted({record["temperature"] for record in records})
    blocks = sorted({int(record["blocks"]) for record in records})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, constrained_layout=True)
    for axis, field, title in zip(
        axes,
        ("train_attention_enrichment", "test_attention_enrichment"),
        ("Training destinations", "Test destinations"),
    ):
        for temperature in temperatures:
            means = []
            stds = []
            plotted_blocks = []
            for block in blocks:
                values = np.asarray(
                    [row[field] for row in records
                     if row["blocks"] == block
                     and row["temperature"] == temperature
                     and row.get(field) is not None],
                    dtype=float,
                )
                if values.size == 0:
                    continue
                plotted_blocks.append(block)
                means.append(np.nanmean(values))
                stds.append(np.nanstd(values))
            if means:
                axis.errorbar(
                    plotted_blocks, means, yerr=stds, marker="o", capsize=4,
                    linewidth=2, label=f"temperature={temperature:g}",
                )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="uniform baseline")
        axis.set_xlabel("Number of applied GAT blocks")
        axis.set_ylabel("Observed / uniform same-label attention")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("GAT attention enrichment over graph composition")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "attention_enrichment_vs_gat_blocks.png")
    plt.close(fig)


def _plot_attention_diagnostics(
    records: list[dict], output_dir: Path, attention_top_k: list[int]
) -> None:
    """Plot entropy and top-k same-label accuracy for each temperature."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not records:
        return
    temperatures = sorted({record["temperature"] for record in records})
    blocks = sorted({int(record["blocks"]) for record in records})
    top_k_values = sorted(set(attention_top_k))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=150, constrained_layout=True)
    for axis, split in zip(axes, ("train", "test")):
        entropy_field = f"{split}_attention_entropy"
        for temperature in temperatures:
            values = []
            plotted_blocks = []
            for block in blocks:
                current = np.asarray(
                    [row[entropy_field] for row in records
                     if row["blocks"] == block
                     and row["temperature"] == temperature
                     and row.get(entropy_field) is not None],
                    dtype=float,
                )
                if current.size:
                    plotted_blocks.append(block)
                    values.append(np.nanmean(current))
            if values:
                axis.plot(
                    plotted_blocks, values, marker="o", linewidth=2,
                    label=f"entropy, T={temperature:g}",
                )
        for k in top_k_values:
            field = f"{split}_top_{k}_same_label_accuracy"
            for temperature in temperatures:
                values = []
                plotted_blocks = []
                for block in blocks:
                    current = np.asarray(
                        [row[field] for row in records
                         if row["blocks"] == block
                         and row["temperature"] == temperature
                         and row.get(field) is not None],
                        dtype=float,
                    )
                    if current.size:
                        plotted_blocks.append(block)
                        values.append(np.nanmean(current))
                if values:
                    axis.plot(
                        plotted_blocks, values, marker="x", linestyle="--",
                        linewidth=1.5, label=f"top-{k}, T={temperature:g}",
                    )
        axis.set_xlabel("Number of applied GAT blocks")
        axis.set_ylabel("Entropy / same-label top-k accuracy")
        axis.set_title(f"{split.capitalize()} attention diagnostics")
        axis.set_xticks(blocks)
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    fig.suptitle("GAT attention entropy and top-k label accuracy")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "attention_diagnostics_vs_gat_blocks.png")
    plt.close(fig)


def _plot_graph_ablation(rows: list[dict], attention_rows: list[dict], output_dir: Path) -> None:
    """Plot prediction and enrichment results for controlled graph topologies."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not rows:
        return
    variants = ["empty", "fully_connected", "cross_label_70", "cross_label_10"]
    labels = ["empty", "fully connected", "70% cross-label", "10% cross-label"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
    for axis, field, title, ylabel in zip(
        axes[:2],
        ("cross_entropy", "balanced_accuracy"),
        ("Cross-entropy", "Balanced accuracy"),
        ("Cross-entropy (lower is better)", "Balanced accuracy (higher is better)"),
    ):
        values = []
        errors = []
        for variant in variants:
            current = np.asarray(
                [row[field] for row in rows if row["graph_ablation"] == variant and row.get(field) is not None],
                dtype=float,
            )
            values.append(np.nanmean(current) if current.size else np.nan)
            errors.append(np.nanstd(current) if current.size else 0.0)
        axis.errorbar(range(len(variants)), values, yerr=errors, fmt="o", capsize=4, linewidth=2)
        axis.set_xticks(range(len(variants)), labels, rotation=20)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    fig.suptitle("Graph topology ablation: prediction metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "graph_ablation_metrics.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), dpi=150, constrained_layout=True)
    values = []
    errors = []
    for variant in variants:
        current = np.asarray(
            [row["test_attention_enrichment"] for row in attention_rows
             if row["graph_ablation"] == variant
             and row.get("test_attention_enrichment") is not None],
            dtype=float,
        )
        values.append(np.nanmean(current) if current.size else np.nan)
        errors.append(np.nanstd(current) if current.size else 0.0)
    axis.errorbar(range(len(variants)), values, yerr=errors, fmt="o", capsize=4, linewidth=2)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="uniform baseline")
    axis.set_xticks(range(len(variants)), labels, rotation=20)
    axis.set_title("Graph topology ablation: test enrichment")
    axis.set_ylabel("Observed / uniform")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "graph_ablation_enrichment.png")
    plt.close(fig)


def main() -> None:
    args = _parser().parse_args()
    if args.num_datasets < 1 or args.num_classes < 2:
        raise ValueError("num-datasets must be positive and num-classes must be at least 2")
    if args.n_estimators < 1:
        raise ValueError("n-estimators must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    temperatures = sorted(set(args.temperatures))
    if not temperatures or any(not np.isfinite(value) or value <= 0 for value in temperatures):
        raise ValueError(f"Temperatures must be finite and positive, got {args.temperatures}")
    attention_top_k = sorted(set(args.attention_top_k))
    if not attention_top_k or any(k <= 0 for k in attention_top_k):
        raise ValueError(f"Attention top-k values must be positive, got {args.attention_top_k}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.show_graph and args.num_graphs != 1:
        raise ValueError("--show-graph requires --num-graphs 1")

    prior = PriorDataset(
        batch_size=args.batch_size,
        batch_size_per_gp=args.batch_size_per_gp,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.num_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        min_train_size=args.min_train_size,
        max_train_size=args.max_train_size,
        prior_type=args.prior_type,
        device=args.prior_device,
        n_jobs=1,
        normalization=args.normalization,
    )
    rows: list[dict] = []
    attention_rows: list[dict] = []
    graph_ablation_rows: list[dict] = []
    graph_ablation_attention_rows: list[dict] = []
    graph_config = {"graph_num_graphs": args.num_graphs} if args.num_graphs is not None else None
    graph_edges: tuple[np.ndarray, np.ndarray] | None = None
    processed = 0
    while processed < args.num_datasets:
        batch = _regular_batch(prior.get_batch())
        for index in range(int(batch[0].shape[0])):
            if processed >= args.num_datasets:
                break
            graph_edges = None
            seq_len = int(batch[3][index].item())
            train_size = int(batch[4][index].item())
            labels = batch[1][index, :seq_len].numpy().astype(int)
            x = batch[0][index, :seq_len, : int(batch[2][index].item())].numpy()
            y_train, y_test = labels[:train_size], labels[train_size:]
            if not np.isin(np.unique(y_test), np.unique(y_train)).all():
                processed += 1
                continue

            skip = _build_classifier(
                args.checkpoint, device, args.seed + processed,
                skip_gat=True, graph_config=graph_config,
            )
            skip.fit(x[:train_size], y_train)
            skip_proba = skip.predict_proba(x[train_size:])
            skip_repr = skip.predict_representation(x[train_size:])
            skip_labels = skip.classes_
            skip_pred = skip_labels[np.argmax(skip_proba, axis=1)]
            rows.append({
                "dataset": processed, "variant": "skip_gat", "blocks": 0,
                "temperature": None,
                "cross_entropy": float(log_loss(y_test, skip_proba, labels=skip_labels)),
                "balanced_accuracy": _balanced_accuracy(y_test, skip_pred),
            })

            encoder_representations: dict[str, np.ndarray] = {}
            if args.encoder_checkpoint is not None:
                encoder = _build_encoder_classifier(
                    args.encoder_checkpoint,
                    device,
                    args.seed + processed + 100000,
                    args.n_estimators,
                )
                encoder.fit(x[:train_size], y_train)
                encoder_proba = encoder.predict_proba(x[train_size:])
                encoder_representations = _encoder_layer_representations(encoder, x)
                encoder_pred = encoder.classes_[np.argmax(encoder_proba, axis=1)]
                rows.append({
                    "dataset": processed, "variant": "encoder", "blocks": -1,
                    "temperature": None,
                    "cross_entropy": float(log_loss(
                        y_test, encoder_proba, labels=encoder.classes_
                    )),
                    "balanced_accuracy": _balanced_accuracy(y_test, encoder_pred),
                })

            if args.graph_ablation:
                ablation_names = ("empty", "fully_connected", "cross_label_70", "cross_label_10")
                for ablation_index, ablation in enumerate(ablation_names):
                    ablation_config = dict(graph_config or {})
                    ablation_config["graph_ablation"] = ablation
                    classifier = _build_classifier(
                        args.checkpoint, device,
                        args.seed + processed + 1000 + ablation_index,
                        graph_config=ablation_config,
                    )
                    classifier.fit(x[:train_size], y_train)
                    _set_temperature(classifier, args.graph_ablation_temperature)
                    prefixes = _prefixes(classifier, args.blocks)
                    prefix = max(prefixes)
                    if prefix == 0:
                        raise ValueError("Graph ablation requires at least one GAT block")
                    gat = classifier.model_.model.icl_predictor.gat_icl
                    original_blocks = gat.graph_blocks
                    try:
                        gat.graph_blocks = torch.nn.ModuleList(list(original_blocks)[:prefix])
                        probabilities = classifier.predict_proba(x[train_size:])
                        representation = classifier.predict_representation(x[train_size:])
                        predicted = classifier.classes_[np.argmax(probabilities, axis=1)]
                        graph_ablation_rows.append({
                            "dataset": processed,
                            "graph_ablation": ablation,
                            "blocks": prefix,
                            "temperature": args.graph_ablation_temperature,
                            "cross_entropy": float(log_loss(
                                y_test, probabilities, labels=classifier.classes_
                            )),
                            "balanced_accuracy": _balanced_accuracy(y_test, predicted),
                        })
                        for record in _attention_statistics(
                            classifier.model_, labels, train_size, attention_top_k
                        ):
                            graph_ablation_attention_rows.append({
                                "dataset": processed,
                                "graph_ablation": ablation,
                                "blocks": prefix,
                                "temperature": args.graph_ablation_temperature,
                                **record,
                            })
                    finally:
                        gat.graph_blocks = original_blocks

            for temperature in temperatures:
                classifier = _build_classifier(
                    args.checkpoint, device, args.seed + processed,
                    graph_config=graph_config,
                )
                classifier.fit(x[:train_size], y_train)
                _set_temperature(classifier, temperature)
                prefixes = _prefixes(classifier, args.blocks)
                full_repr: dict[str, np.ndarray] = {"skip_gat": skip_repr}
                full_repr.update(encoder_representations)
                for prefix in prefixes:
                    if prefix == 0:
                        continue
                    proba, representation, attention = _run_prefix(
                        classifier, x[:train_size], y_train, x[train_size:],
                        prefix, labels, temperature, attention_top_k,
                    )
                    variant = f"gat_{prefix}"
                    full_repr[variant] = representation
                    predicted = classifier.classes_[np.argmax(proba, axis=1)]
                    rows.append({
                        "dataset": processed, "variant": variant, "blocks": prefix,
                        "temperature": temperature,
                        "cross_entropy": float(log_loss(y_test, proba, labels=classifier.classes_)),
                        "balanced_accuracy": _balanced_accuracy(y_test, predicted),
                    })
                    for record in attention:
                        attention_rows.append({
                            "dataset": processed, "blocks": prefix,
                            "temperature": temperature, **record,
                        })
                    if args.show_graph and graph_edges is None:
                        graph_edges = _cached_graph_edges(classifier)

                _plot_umap(
                    full_repr, labels, train_size,
                    args.output_dir / "umap" / f"temperature_{temperature:g}"
                    / f"dataset{processed:05d}.png",
                    args.seed + processed, args.umap_n_neighbors, args.umap_n_epochs,
                    graph_edges=graph_edges if args.show_graph else None,
                )
            processed += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    (args.output_dir / "attention_statistics.json").write_text(json.dumps(attention_rows, indent=2))
    if args.graph_ablation:
        (args.output_dir / "graph_ablation_metrics.json").write_text(
            json.dumps(graph_ablation_rows, indent=2)
        )
        (args.output_dir / "graph_ablation_attention_statistics.json").write_text(
            json.dumps(graph_ablation_attention_rows, indent=2)
        )
    _plot_metrics(rows, args.output_dir)
    _plot_attention_statistics(attention_rows, args.output_dir)
    _plot_attention_enrichment(attention_rows, args.output_dir)
    _plot_attention_diagnostics(attention_rows, args.output_dir, attention_top_k)
    if args.graph_ablation:
        _plot_graph_ablation(
            graph_ablation_rows, graph_ablation_attention_rows, args.output_dir
        )
    print(f"Saved {len(rows)} metric records to {args.output_dir / 'metrics.json'}")
    print(f"Saved {len(attention_rows)} attention records to {args.output_dir / 'attention_statistics.json'}")


if __name__ == "__main__":
    main()
