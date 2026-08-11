"""Compare legacy and PyG graph attention on the same inputs.

Example: ``python scripts/gat_vs_gat_pyg_bench.py --device cpu --repeats 3``.
"""
from __future__ import annotations

import argparse
import time

import torch

from tabicl._model.gat import GraphAttentionTransformer
from tabicl._model.gat_pyg import GraphAttentionTransformer as PyGGraphAttentionTransformer


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=32_768)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--edges-per-node", type=int, default=3)
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
        source = torch.randn(
            args.batch_size, args.samples, args.features, args.d_model,
            device=device, dtype=dtype,
        )
        model_args = dict(
            num_blocks=args.layers,
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_graphs=args.num_graphs,
            max_chunk_size=args.edge_block_size,
        )
        legacy = GraphAttentionTransformer(**model_args).to(device=device, dtype=dtype).eval()
        pyg = PyGGraphAttentionTransformer(**model_args).to(device=device, dtype=dtype).eval()
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
        with torch.inference_mode():
            for _ in range(2):
                model(source, edge_index_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.repeats):
                output = model(source, edge_index_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return (time.perf_counter() - start) / args.repeats, output

    try:
        legacy_time, expected = measure(legacy)
    except BaseException as error:
        if not is_oom_error(error):
            raise
        report_oom("legacy", error, device)
        legacy_time = None
        expected = None

    # Release any cached legacy temporaries before measuring the PyG model.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        pyg_time, actual = measure(pyg)
    except BaseException as error:
        if not is_oom_error(error):
            raise
        report_oom("pyg", error, device)
        pyg_time = None
        actual = None

    if legacy_time is not None:
        print(f"legacy_seconds={legacy_time:.6f}")
    if pyg_time is not None:
        print(f"pyg_seconds={pyg_time:.6f}")
    if legacy_time is not None and pyg_time is not None:
        print(f"speedup={legacy_time / pyg_time:.3f}")
    if actual is not None and expected is not None:
        print(f"max_abs_error={(actual - expected).abs().max().item():.6g}")
    if device.type == "cuda":
        print(f"pyg_peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 2**20:.1f}")
        print(f"pyg_peak_reserved_mb={torch.cuda.max_memory_reserved(device) / 2**20:.1f}")


if __name__ == "__main__":
    main()
