"""Pure-load benchmark for a PyG implementation of graph multi-head attention.

This script intentionally does not modify or benchmark ``gat.py``'s
implementation.  It provides a PyTorch Geometric implementation with the same
scaled dot-product, destination-wise softmax semantics and measures inference
(memory/load) only: no gradients, optimizer state, or backward pass.

The benchmark writes two figures:

* ``edges_vs_samples.png``: total graph edges versus samples/nodes;
* ``edges_vs_features.png``: feature-expanded message edges versus features.

Successful points are green and OOM points are red.  The feature sweep uses
``base_edges * n_features`` on the x-axis because the operation applies the
same graph independently to every feature column.

Example::

    python scripts/test_pyg_attention.py --device cuda --output-dir results/pyg_attention
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


FEATURE_BLOCK_SIZE = 256
EDGE_BLOCK_SIZE = 12_000_000


@dataclass
class BenchmarkResult:
    sweep: str
    samples: int
    features: int
    base_edges: int
    expanded_edges: int
    succeeded: bool
    seconds: float | None = None
    process_memory_before_mb: float | None = None
    process_memory_after_mb: float | None = None
    peak_process_memory_mb: float | None = None
    peak_cuda_allocated_mb: float | None = None
    peak_cuda_reserved_mb: float | None = None
    forward_seconds: float | None = None
    feature_blocks: int = 0
    feature_block_seconds: float | None = None
    feature_block_mean_seconds: float | None = None
    edge_blocks: int = 0
    edge_block_build_seconds: float | None = None
    edge_block_seconds: float | None = None
    edge_block_mean_seconds: float | None = None
    sorting_seconds: float | None = None
    linear_projection_seconds: float | None = None
    linear_projection_mean_seconds: float | None = None
    edge_loop_seconds: float | None = None
    edge_loop_mean_seconds: float | None = None
    error: str | None = None


@dataclass
class ForwardTiming:
    """Timing counters collected during one forward pass."""

    feature_blocks: int = 0
    feature_block_seconds: float = 0.0
    edge_blocks: int = 0
    edge_block_build_seconds: float = 0.0
    edge_block_seconds: float = 0.0
    sorting_seconds: float = 0.0
    _feature_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list, repr=False)
    _edge_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list, repr=False)
    _projection_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list, repr=False)
    _edge_loop_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list, repr=False)
    linear_projection_seconds: float = 0.0
    edge_loop_seconds: float = 0.0

    def finalize_cuda(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        self.feature_block_seconds = sum(
            start.elapsed_time(end) for start, end in self._feature_events
        ) / 1000.0
        self.edge_block_seconds = sum(
            start.elapsed_time(end) for start, end in self._edge_events
        ) / 1000.0
        self.linear_projection_seconds = sum(
            start.elapsed_time(end) for start, end in self._projection_events
        ) / 1000.0
        self.edge_loop_seconds = sum(
            start.elapsed_time(end) for start, end in self._edge_loop_events
        ) / 1000.0


def _cuda_timer(device: torch.device) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
    if device.type != "cuda":
        return None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


class ProcessMemorySampler:
    """Sample RSS because PyTorch CPU allocations are not tracked by tracemalloc."""

    def __init__(self, interval: float = 0.005) -> None:
        self.process = psutil.Process()
        self.interval = interval
        self.before_mb = self._rss_mb()
        self.peak_mb = self.before_mb
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _rss_mb(self) -> float:
        return self.process.memory_info().rss / 2**20

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_mb = max(self.peak_mb, self._rss_mb())

    def __enter__(self) -> "ProcessMemorySampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.after_mb = self._rss_mb()
        self.peak_mb = max(self.peak_mb, self.after_mb)


class PyGGraphMultiheadAttention(MessagePassing):
    """Memory-conscious PyG equivalent of ``GraphMultiheadAttention``.

    Nodes are flattened as independent ``(dataset, feature-column)`` graph
    copies.  PyG performs destination-indexed softmax and aggregation through
    its message-passing/scatter implementation rather than materializing the
    dense ``(destination, source, feature, head)`` attention tensor.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        learnable_residual: bool = False,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.learnable_residual = bool(learnable_residual)
        self._sorted_group_cache: dict[tuple[int, int, int, str], tuple[Tensor, list[int]]] = {}
        if self.learnable_residual:
            self.alpha = nn.Parameter(torch.logit(torch.tensor(0.2, dtype=torch.float32)))

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        index: Tensor,
        ptr: Tensor | None,
        size_i: int | None,
    ) -> Tensor:
        logits = (q_i * k_j).sum(dim=-1) * self.scale
        weights = softmax(logits, index=index, ptr=ptr, num_nodes=size_i)
        weights = self.dropout(weights)
        return v_j * weights.unsqueeze(-1)

    @staticmethod
    def _edge_blocks(
        edge_index_batch: Iterable[Tensor],
        columns: int,
        nodes: int,
        device: torch.device,
        max_edges: int,
        timing: ForwardTiming | None = None,
        sorted_group_cache: dict[tuple[int, int, int, str], tuple[Tensor, list[int]]] | None = None,
    ) -> Iterator[Tensor]:
        """Yield edge blocks without splitting destination softmax groups."""
        if max_edges <= 0:
            raise ValueError("max_edges must be positive")

        # Each descriptor represents a contiguous range of destination groups
        # from one graph copy.  Keeping descriptors instead of one tensor per
        # group lets the actual expansion below use one vectorized slice/add
        # operation for the complete range.
        block_ranges: list[tuple[Tensor, int, int, int]] = []
        block_size = 0

        def emit_block() -> Tensor:
            build_started = time.perf_counter()
            edge_block = torch.cat(
                [
                    sorted_edges[:, start:end] + offset
                    for sorted_edges, start, end, offset in block_ranges
                ],
                dim=1,
            )
            if timing is not None:
                timing.edge_block_build_seconds += time.perf_counter() - build_started
            return edge_block

        for batch_index, edge_index in enumerate(edge_index_batch):
            source_edge_index = edge_index
            edge_index = edge_index.to(device=device, dtype=torch.long)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("Each edge index must have shape (2, E)")
            if edge_index.numel() == 0:
                continue

            cache_key = (
                id(source_edge_index),
                source_edge_index.data_ptr(),
                source_edge_index.shape[1],
                str(device),
            )
            cached = sorted_group_cache.get(cache_key) if sorted_group_cache is not None else None
            if cached is None:
                # Sorting once per input graph makes every destination group
                # contiguous. Each yielded block can therefore use PyG's
                # regular destination-wise softmax without changing results.
                sort_started = time.perf_counter()
                order = torch.argsort(edge_index[1], stable=True)
                sorted_edges = edge_index[:, order]
                group_starts = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long, device=device),
                        torch.where(sorted_edges[1, 1:] != sorted_edges[1, :-1])[0].add(1),
                        torch.tensor([sorted_edges.shape[1]], dtype=torch.long, device=device),
                    ]
                ).tolist()
                group_lengths = [
                    end - start for start, end in zip(group_starts[:-1], group_starts[1:])
                ]
                if sorted_group_cache is not None:
                    sorted_group_cache[cache_key] = (sorted_edges, group_lengths)
                if timing is not None:
                    timing.sorting_seconds += time.perf_counter() - sort_started
            else:
                sorted_edges, group_lengths = cached

            group_offsets = [0]
            group_offsets.extend(
                group_offsets[-1] + length for length in group_lengths
            )

            for column in range(columns):
                offset = (batch_index * columns + column) * nodes
                copy_group_start = 0
                for group_edges in group_lengths:
                    if block_ranges and block_size + group_edges > max_edges:
                        yield emit_block()
                        block_ranges = []
                        block_size = 0
                    # Extend the current copy range rather than constructing
                    # and storing a tensor for each destination group.
                    copy_group_end = copy_group_start + 1
                    if block_ranges and block_ranges[-1][0] is sorted_edges and block_ranges[-1][3] == offset:
                        previous_edges, previous_start, previous_end, previous_offset = block_ranges[-1]
                        if previous_end == group_offsets[copy_group_start]:
                            block_ranges[-1] = (
                                previous_edges,
                                previous_start,
                                group_offsets[copy_group_end],
                                previous_offset,
                            )
                        else:
                            block_ranges.append((sorted_edges, group_offsets[copy_group_start], group_offsets[copy_group_end], offset))
                    else:
                        block_ranges.append((sorted_edges, group_offsets[copy_group_start], group_offsets[copy_group_end], offset))
                    block_size += group_edges
                    copy_group_start = copy_group_end
                    # A single destination group cannot be split without
                    # changing its softmax. It is yielded as an oversized
                    # block when it exceeds max_edges by itself.
                    if block_size >= max_edges:
                        yield emit_block()
                        block_ranges = []
                        block_size = 0

        if block_ranges:
            yield emit_block()

    def forward(
        self,
        src: Tensor,
        edge_index_batch: list[Tensor],
        residual_src: Tensor | None = None,
        feature_block_size: int = FEATURE_BLOCK_SIZE,
        edge_block_size: int = EDGE_BLOCK_SIZE,
        timing: ForwardTiming | None = None,
    ) -> Tensor:
        if src.ndim != 4:
            raise ValueError("src must have shape (B, T, C, D)")
        if len(edge_index_batch) != src.shape[0]:
            raise ValueError("edge_index_batch length must equal batch size")
        if residual_src is None:
            residual_src = src
        if residual_src.shape != src.shape:
            raise ValueError("residual_src must have the same shape as src")
        if feature_block_size <= 0:
            raise ValueError("feature_block_size must be positive")
        if edge_block_size <= 0:
            raise ValueError("edge_block_size must be positive")

        batch_size, nodes, columns, _ = src.shape
        # Each feature column is an independent graph copy. Process columns in
        # bounded blocks so the expanded edge index and message tensors scale
        # with ``feature_block_size`` rather than with all columns at once.
        # Preallocate the complete result once. Keeping a list of block
        # outputs until torch.cat() would retain every block simultaneously.
        output = torch.empty_like(src)
        for start in range(0, columns, feature_block_size):
            feature_started = time.perf_counter()
            feature_events = _cuda_timer(src.device)
            if feature_events is not None:
                feature_events[0].record()
            end = min(start + feature_block_size, columns)
            block_columns = end - start
            src_block = src[:, :, start:end, :]
            x = src_block.permute(0, 2, 1, 3).contiguous().reshape(
                batch_size * block_columns * nodes, self.d_model
            )
            aggregated = torch.zeros_like(x).view(-1, self.nhead, self.head_dim)
            projection_started = time.perf_counter()
            projection_events = _cuda_timer(src.device)
            if projection_events is not None:
                projection_events[0].record()
            qkv = self.qkv_proj(x).view(-1, 3, self.nhead, self.head_dim)
            q, k, v = qkv.unbind(dim=1)
            if projection_events is not None:
                projection_events[1].record()
            elif timing is not None:
                timing.linear_projection_seconds += time.perf_counter() - projection_started
            if timing is not None and projection_events is not None:
                timing._projection_events.append(projection_events)

            edge_loop_started = time.perf_counter()
            edge_loop_events = _cuda_timer(src.device)
            if edge_loop_events is not None:
                edge_loop_events[0].record()
            for edge_index in self._edge_blocks(
                edge_index_batch,
                block_columns,
                nodes,
                src.device,
                edge_block_size,
                timing,
                self._sorted_group_cache,
            ):
                edge_started = time.perf_counter()
                edge_events = _cuda_timer(src.device)
                if edge_events is not None:
                    edge_events[0].record()
                aggregated.add_(self.propagate(edge_index, q=q, k=k, v=v, size=(x.shape[0], x.shape[0])))
                if timing is not None:
                    timing.edge_blocks += 1
                    if edge_events is not None:
                        edge_events[1].record()
                        timing._edge_events.append(edge_events)
                    else:
                        timing.edge_block_seconds += time.perf_counter() - edge_started
            if edge_loop_events is not None:
                edge_loop_events[1].record()
            elif timing is not None:
                timing.edge_loop_seconds += time.perf_counter() - edge_loop_started
            if timing is not None and edge_loop_events is not None:
                timing._edge_loop_events.append(edge_loop_events)
            block_output = self.out_proj(aggregated.reshape(-1, self.d_model))
            block_output = block_output.reshape(batch_size, block_columns, nodes, self.d_model).permute(0, 2, 1, 3)
            output[:, :, start:end, :] = block_output
            if timing is not None:
                timing.feature_blocks += 1
                if feature_events is not None:
                    feature_events[1].record()
                    timing._feature_events.append(feature_events)
                else:
                    timing.feature_block_seconds += time.perf_counter() - feature_started

        if self.learnable_residual:
            alpha = torch.sigmoid(self.alpha).to(dtype=src.dtype)
            return (1.0 - alpha) * residual_src + alpha * output
        return residual_src + output


def make_edges(nodes: int, edges: int, seed: int = 0) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randint(nodes, (edges,), generator=generator)
    destination = torch.randint(nodes, (edges,), generator=generator)
    return torch.stack((source, destination))


def is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, (MemoryError,)) or "out of memory" in text or "cuda error" in text and "memory" in text


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_point(
    *,
    sweep: str,
    samples: int,
    features: int,
    edges: int,
    batch_size: int,
    d_model: int,
    nhead: int,
    feature_block_size: int,
    edge_block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> BenchmarkResult:
    expanded_edges = edges * batch_size * features
    result = BenchmarkResult(sweep, samples, features, edges, expanded_edges, False)
    with ProcessMemorySampler() as memory:
        try:
            torch.manual_seed(seed)
            edge_index = make_edges(samples, edges, seed=seed)
            edge_batch = [edge_index for _ in range(batch_size)]
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            src = torch.randn(batch_size, samples, features, d_model, device=device, dtype=dtype)
            layer = PyGGraphMultiheadAttention(d_model, nhead).to(device=device, dtype=dtype).eval()
            synchronize(device)
            timing = ForwardTiming()
            started = time.perf_counter()
            total_events = _cuda_timer(device)
            if total_events is not None:
                total_events[0].record()
            with torch.inference_mode():
                output = layer(
                    src,
                    edge_batch,
                    feature_block_size=feature_block_size,
                    edge_block_size=edge_block_size,
                    timing=timing,
                )
            if total_events is not None:
                total_events[1].record()
            synchronize(device)
            timing.finalize_cuda(device)
            result.seconds = (
                total_events[0].elapsed_time(total_events[1]) / 1000.0
                if total_events is not None
                else time.perf_counter() - started
            )
            result.forward_seconds = result.seconds
            result.feature_blocks = timing.feature_blocks
            result.feature_block_seconds = timing.feature_block_seconds
            result.feature_block_mean_seconds = (
                timing.feature_block_seconds / timing.feature_blocks if timing.feature_blocks else None
            )
            result.edge_blocks = timing.edge_blocks
            result.edge_block_build_seconds = timing.edge_block_build_seconds
            result.edge_block_seconds = timing.edge_block_seconds
            result.edge_block_mean_seconds = (
                timing.edge_block_seconds / timing.edge_blocks if timing.edge_blocks else None
            )
            result.sorting_seconds = timing.sorting_seconds
            result.linear_projection_seconds = timing.linear_projection_seconds
            result.linear_projection_mean_seconds = (
                timing.linear_projection_seconds / timing.feature_blocks
                if timing.feature_blocks else None
            )
            result.edge_loop_seconds = timing.edge_loop_seconds
            result.edge_loop_mean_seconds = (
                timing.edge_loop_seconds / timing.feature_blocks
                if timing.feature_blocks else None
            )
            result.succeeded = tuple(output.shape) == (batch_size, samples, features, d_model)
            if device.type == "cuda":
                result.peak_cuda_allocated_mb = torch.cuda.max_memory_allocated(device) / 2**20
                result.peak_cuda_reserved_mb = torch.cuda.max_memory_reserved(device) / 2**20
        except BaseException as error:  # benchmark points must continue after OOM
            result.error = f"{type(error).__name__}: {error}"[:500]
            result.succeeded = False
            if not is_oom(error):
                print(f"{sweep}: unexpected failure at samples={samples}, features={features}: {result.error}")
        finally:
            for name in ("output", "src", "layer"):
                if name in locals():
                    del locals()[name]
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result.process_memory_before_mb = memory.before_mb
    result.process_memory_after_mb = memory.after_mb
    result.peak_process_memory_mb = memory.peak_mb
    return result


def parse_grid(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("grid must contain positive comma-separated integers")
    return values


def plot_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sweep, x_label, y_label, path in (
        ("samples", "base graph edges", "number of samples / nodes", "edges_vs_samples.png"),
        ("features", "feature-expanded message edges", "number of features", "edges_vs_features.png"),
    ):
        points = [item for item in results if item.sweep == sweep]
        successful = [item for item in points if item.succeeded]
        failed = [item for item in points if not item.succeeded]
        figure, axis = plt.subplots(figsize=(8, 5))
        if successful:
            axis.scatter(
                [item.base_edges if sweep == "samples" else item.expanded_edges for item in successful],
                [item.samples if sweep == "samples" else item.features for item in successful],
                color="tab:green", label="successful", marker="o",
            )
        if failed:
            axis.scatter(
                [item.base_edges if sweep == "samples" else item.expanded_edges for item in failed],
                [item.samples if sweep == "samples" else item.features for item in failed],
                color="tab:red", label="OOM / failed", marker="x", s=70,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(f"PyG graph attention pure-load sweep: edges vs {sweep}")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / path, dpi=160)
        plt.close(figure)


def check_semantics(device: torch.device, dtype: torch.dtype) -> None:
    """Compare the PyG layer with the repository layer on a small eval graph."""
    from tabicl._model.gat import GraphMultiheadAttention

    torch.manual_seed(123)
    reference = GraphMultiheadAttention(d_model=8, nhead=2, dropout=0.0).to(device=device, dtype=dtype).eval()
    candidate = PyGGraphMultiheadAttention(d_model=8, nhead=2, dropout=0.0).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        candidate.qkv_proj.weight.copy_(torch.cat(
            [reference.q_proj.weight, reference.k_proj.weight, reference.v_proj.weight], dim=0
        ))
        candidate.qkv_proj.bias.copy_(torch.cat(
            [reference.q_proj.bias, reference.k_proj.bias, reference.v_proj.bias], dim=0
        ))
        candidate.out_proj.load_state_dict(reference.out_proj.state_dict())
    source = torch.randn(2, 5, 3, 8, device=device, dtype=dtype)
    edges = [make_edges(5, 12, 99).to(device), make_edges(5, 12, 100).to(device)]
    with torch.inference_mode():
        expected = reference(source, edges)
        actual = candidate(source, edges, feature_block_size=2, edge_block_size=3)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    print("semantic parity check: passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results/pyg_attention"))
    parser.add_argument("--sample-grid", type=parse_grid, default=parse_grid("1024,2048,"))
    parser.add_argument("--feature-grid", type=parse_grid, default=parse_grid("32,64,128,512,1024"))
    parser.add_argument("--samples-for-feature-sweep", type=int, default=65536)
    parser.add_argument("--features-for-sample-sweep", type=int, default=1024)
    parser.add_argument("--edges-per-node", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--feature-block-size", type=int, default=FEATURE_BLOCK_SIZE)
    parser.add_argument("--edge-block-size", type=int, default=EDGE_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-parity-check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if (
        args.edges_per_node <= 0
        or args.batch_size <= 0
        or args.feature_block_size <= 0
        or args.edge_block_size <= 0
    ):
        raise ValueError("edges-per-node, batch-size, feature-block-size, and edge-block-size must be positive")
    if not args.skip_parity_check:
        check_semantics(device, dtype)

    results: list[BenchmarkResult] = []
    for samples in args.sample_grid:
        print(f"running sample sweep: samples={samples}, features={args.features_for_sample_sweep}")
        results.append(run_point(
            sweep="samples", samples=samples, features=args.features_for_sample_sweep,
            edges=samples * args.edges_per_node, batch_size=args.batch_size,
            d_model=args.d_model, nhead=args.nhead, feature_block_size=args.feature_block_size,
            edge_block_size=args.edge_block_size, dtype=dtype, device=device, seed=args.seed + samples,
        ))
    for features in args.feature_grid:
        print(f"running feature sweep: samples={args.samples_for_feature_sweep}, features={features}")
        results.append(run_point(
            sweep="features", samples=args.samples_for_feature_sweep, features=features,
            edges=args.samples_for_feature_sweep * args.edges_per_node, batch_size=args.batch_size,
            d_model=args.d_model, nhead=args.nhead, feature_block_size=args.feature_block_size,
            edge_block_size=args.edge_block_size, dtype=dtype, device=device, seed=args.seed + features,
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.json").open("w") as handle:
        json.dump([asdict(result) for result in results], handle, indent=2)
    plot_results(results, args.output_dir)
    for result in results:
        status = "ok" if result.succeeded else "OOM/failed"
        print(
            f"{result.sweep:8s} samples={result.samples:6d} features={result.features:4d} "
            f"base_edges={result.base_edges:9d} expanded_edges={result.expanded_edges:12d} "
            f"RSS_peak={result.peak_process_memory_mb:8.1f}MB "
            f"CUDA_alloc={result.peak_cuda_allocated_mb or 0:8.1f}MB {status}"
        )
        if result.succeeded:
            print(
                f"  timing forward={result.forward_seconds:.4f}s "
                f"feature_blocks={result.feature_blocks} "
                f"feature_total={result.feature_block_seconds:.4f}s "
                f"feature_mean={result.feature_block_mean_seconds:.4f}s "
                f"edge_blocks={result.edge_blocks} "
                f"edge_build={result.edge_block_build_seconds:.4f}s "
                f"edge_total={result.edge_block_seconds:.4f}s "
                f"edge_mean={result.edge_block_mean_seconds:.6f}s "
                f"sorting={result.sorting_seconds:.4f}s "
                f"linear_projection={result.linear_projection_seconds:.4f}s "
                f"projection_mean={result.linear_projection_mean_seconds:.4f}s "
                f"edge_loop={result.edge_loop_seconds:.4f}s "
                f"edge_loop_mean={result.edge_loop_mean_seconds:.4f}s"
            )


if __name__ == "__main__":
    main()
