"""Compare legacy and PyG graph attention on the same inputs.

Example: ``python scripts/gat_vs_gat_pyg_bench.py --device cpu --repeats 3``.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import time

import torch

from tabicl._model.gat import Graph1DAttentionTransformer, Graph2DAttentionTransformer
from tabicl._model.gat_pyg import (
    Graph1DAttentionTransformer as PyGGraph1DAttentionTransformer,
    Graph2DAttentionTransformer as PyGGraph2DAttentionTransformer,
)
from tabicl._model.graph import GraphTopologyPrior
from tabicl._model.tabicl import TabICL


def make_edges(nodes: int, edges: int, seed: int = 0) -> torch.Tensor:
    """Generate edges using the same CPU generator as test_pyg_attention."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randint(nodes, (edges,), generator=generator)
    destination = torch.randint(nodes, (edges,), generator=generator)
    return torch.stack((source, destination))


def is_oom_error(error: BaseException) -> bool:
    """Return whether an exception represents a host or CUDA OOM."""
    message = str(error).lower()
    return isinstance(error, MemoryError) or "out of memory" in message or "cannot allocate memory" in message


def report_oom(label: str, error: BaseException, device: torch.device) -> None:
    print(f"{label}_status=OOM")
    print(f"{label}_error={type(error).__name__}: {error}")
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("2d", "1d", "full-compressed"), default="2d")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=32_768)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--edges-per-node", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--num-graphs", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--edge-block-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--training-pass",
        action="store_true",
        help="Benchmark a training pass (forward plus backward) instead of inference only.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if (
        args.edges_per_node <= 0 or args.batch_size <= 0 or args.edge_block_size <= 0
        or args.layers <= 0 or args.num_graphs <= 0
        or args.layers % args.num_graphs != 0
    ):
        raise ValueError("edges-per-node, batch-size, edge-block-size, layers, and num-graphs must be valid")
    torch.manual_seed(args.seed)
    edge_count = args.samples * args.edges_per_node
    edges = make_edges(args.samples, edge_count, seed=args.seed)
    edge_batch = [edges for _ in range(args.batch_size)]
    try:
        if args.mode == "full-compressed":
            source = torch.randn(args.batch_size, args.samples, args.features, device=device, dtype=dtype)
            labels = torch.randint(0, 2, (args.batch_size, args.samples), device=device)
            graph_set = GraphTopologyPrior(
                min_train_neighbors=max(1, min(args.edges_per_node, args.samples - 1)),
                max_train_neighbors=max(1, args.edges_per_node),
                train_neighbors_per_test=max(1, args.edges_per_node),
                seed=args.seed,
            )(labels, n_train=max(1, args.samples // 2), num_graphs=args.num_graphs)
            model_args = dict(
                max_features=args.features,
                max_classes=2,
                embed_dim=args.d_model,
                col_num_blocks=1,
                col_nhead=args.nhead,
                col_num_inds=min(16, args.samples),
                row_num_blocks=1,
                row_nhead=args.nhead,
                row_num_cls=1,
                icl_num_blocks=args.layers,
                icl_nhead=args.nhead,
                graph_num_graphs=args.num_graphs,
                graph_max_chunk_size=args.edge_block_size,
                dropout=0.0,
            )
            # Training mode keeps the benchmark on the requested device; the
            # inference manager may otherwise use its separately configured device.
            legacy = TabICL(**model_args, icl_backend="graph-1d").to(device=device, dtype=dtype).train()
            pyg = TabICL(**model_args, icl_backend="graph-1d-pyg").to(device=device, dtype=dtype).train()
            call_args = dict(y_train=labels[:, : args.samples // 2], graph_set=graph_set)
            measure_source = lambda model: model(source, **call_args)
        else:
            if args.mode == "1d":
                source = torch.randn(args.batch_size, args.samples, args.d_model, device=device, dtype=dtype)
                legacy_cls, pyg_cls = Graph1DAttentionTransformer, PyGGraph1DAttentionTransformer
            else:
                source = torch.randn(
                    args.batch_size, args.samples, args.features, args.d_model,
                    device=device, dtype=dtype,
                )
                legacy_cls, pyg_cls = Graph2DAttentionTransformer, PyGGraph2DAttentionTransformer
            model_args = dict(
                num_blocks=args.layers,
                d_model=args.d_model,
                nhead=args.nhead,
                dim_feedforward=args.dim_feedforward,
                num_graphs=args.num_graphs,
                max_chunk_size=args.edge_block_size,
            )
            legacy = legacy_cls(**model_args).to(device=device, dtype=dtype).eval()
            pyg = pyg_cls(**model_args).to(device=device, dtype=dtype).eval()
            measure_source = lambda model: model(source, edge_index_batch)
        with torch.no_grad():
            pyg.load_state_dict(legacy.state_dict())

        # The transformer consumes one edge list per graph route. Each route
        # contains one local graph per batch item.
        edge_index_batch = [edge_batch for _ in range(args.num_graphs)]
    except BaseException as error:
        if not is_oom_error(error):
            raise
        report_oom("setup", error, device)
        return

    def measure(model):
        # Some embedding layers intentionally cast labels to float32.  When
        # the benchmark model is converted to bfloat16, autocast performs the
        # mixed-precision linear operations without requiring callers to
        # mutate labels or model parameters.
        autocast = (
            torch.autocast(device_type=device.type, dtype=dtype)
            if dtype == torch.bfloat16 and device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        model.train(args.training_pass)
        context = torch.enable_grad() if args.training_pass else torch.inference_mode()
        with context, autocast:
            def run_pass():
                output = measure_source(model)
                if args.training_pass:
                    if isinstance(output, tuple):
                        output = output[0]
                    output.sum().backward()
                    model.zero_grad(set_to_none=True)
                return output

            for _ in range(2):
                run_pass()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.repeats):
                output = run_pass()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            return (time.perf_counter() - start) / args.repeats, output, peak

    try:
        legacy_time, expected, legacy_peak = measure(legacy)
    except BaseException as error:
        if not is_oom_error(error):
            raise
        report_oom("legacy", error, device)
        legacy_time = None
        expected = None
        legacy_peak = None

    # Release any cached legacy temporaries before measuring the PyG model.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        pyg_time, actual, pyg_peak = measure(pyg)
    except BaseException as error:
        if not is_oom_error(error):
            raise
        report_oom("pyg", error, device)
        pyg_time = None
        actual = None
        pyg_peak = None

    if legacy_time is not None:
        print(f"legacy_seconds={legacy_time:.6f}")
    if pyg_time is not None:
        print(f"pyg_seconds={pyg_time:.6f}")
    if legacy_peak is not None:
        print(f"legacy_peak_allocated_mb={legacy_peak / 2**20:.1f}")
    if pyg_peak is not None:
        print(f"pyg_peak_allocated_mb={pyg_peak / 2**20:.1f}")
    if legacy_time is not None and pyg_time is not None:
        print(f"speedup={legacy_time / pyg_time:.3f}")
    if actual is not None and expected is not None:
        print(f"max_abs_error={(actual - expected).abs().max().item():.6g}")
    if device.type == "cuda":
        print(f"pyg_peak_reserved_mb={torch.cuda.max_memory_reserved(device) / 2**20:.1f}")


if __name__ == "__main__":
    main()
