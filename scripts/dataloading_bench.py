"""Benchmark synthetic prior data-loading throughput.

This follows the on-the-fly prior setup used by ``scripts/train_stage1.sh``.
The prior already yields complete batches, so the outer ``DataLoader`` uses
``batch_size=None``; ``--batch-sizes`` controls ``PriorDataset.batch_size``.

Example::

	python scripts/dataloading_bench.py \
		--num-workers 0,2,4,8 --batch-sizes 64,256,512 --num-batches 20
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tabicl.prior._dataset import PriorDataset


GRAPH_BACKENDS = {"graph", "graph-pyg", "graph-2d", "graph-2d-pyg", "graph-1d", "graph-1d-pyg"}


@dataclass
class Result:
	num_workers: int
	batch_size: int
	startup_seconds: float
	first_batch_seconds: float
	mean_batch_seconds: float
	median_batch_seconds: float
	batches_measured: int
	samples_per_second: float


def parse_int_list(value: str) -> list[int]:
	"""Parse a comma-separated list of positive integers."""
	values = [int(item.strip()) for item in value.split(",") if item.strip()]
	if not values or any(item < 0 for item in values):
		raise argparse.ArgumentTypeError("expected a comma-separated list of non-negative integers")
	return values


def shutdown_loader(iterator) -> None:
	"""Stop persistent workers before creating the next benchmark case."""
	shutdown = getattr(iterator, "_shutdown_workers", None)
	if shutdown is not None:
		shutdown()


def benchmark_case(args: argparse.Namespace, num_workers: int, batch_size: int) -> Result:
	graph_backend = args.icl_backend in GRAPH_BACKENDS
	dataset = PriorDataset(
		batch_size=batch_size,
		batch_size_per_gp=batch_size,
		batch_size_per_subgp=batch_size,
		min_features=args.min_features,
		max_features=args.max_features,
		max_classes=args.max_classes,
		min_seq_len=args.min_seq_len,
		max_seq_len=args.max_seq_len,
		min_train_size=args.min_train_size,
		max_train_size=args.max_train_size,
		prior_type=args.prior_type,
		device=args.prior_device,
		n_jobs=1,
		num_threads_per_generate=args.num_threads_per_generate,
		normalization=args.normalization,
		graph_backend=graph_backend,
		graph_num_graphs=args.graph_num_graphs,
		graph_min_train_neighbors=args.graph_min_train_neighbors,
		graph_max_train_neighbors=args.graph_max_train_neighbors,
		graph_cross_label_fraction=args.graph_cross_label_fraction,
		graph_train_neighbors_per_test=args.graph_train_neighbors_per_test,
		graph_seed=args.seed,
		graph_v1_prob=args.graph_v1_prob,
		graph_v2_prob=args.graph_v2_prob,
		graph_prob=args.graph_prob,
	)
	loader_kwargs = {
		"batch_size": None,
		"num_workers": num_workers,
		"pin_memory": args.pin_memory,
		"persistent_workers": num_workers > 0,
		"in_order": False,
	}
	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = args.prefetch_factor

	start = time.perf_counter()
	loader = DataLoader(dataset, **loader_kwargs)
	iterator = iter(loader)
	first_batch = next(iterator)
	startup = time.perf_counter() - start

	batch_times = []
	for _ in range(args.num_batches):
		print(f"  measuring batch {_ + 1}/{args.num_batches} ...", flush=True)
		batch_start = time.perf_counter()
		next(iterator)
		batch_times.append(time.perf_counter() - batch_start)
		print("  done", flush=True)

	try:
		shutdown_loader(iterator)
	finally:
		del iterator, loader, dataset, first_batch
		gc.collect()

	batch_times_tensor = np.asarray(batch_times, dtype=np.float64)
	mean_batch = float(batch_times_tensor.mean())
	return Result(
		num_workers=num_workers,
		batch_size=batch_size,
		startup_seconds=startup,
		first_batch_seconds=startup,
		mean_batch_seconds=mean_batch,
		median_batch_seconds=float(np.median(batch_times_tensor)),
		batches_measured=len(batch_times),
		samples_per_second=batch_size / mean_batch if mean_batch > 0 else float("inf"),
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--num-workers", type=parse_int_list, default=[0, 2, 4, 8])
	parser.add_argument("--batch-sizes", type=parse_int_list, default=[64, 256, 512])
	parser.add_argument("--num-batches", type=int, default=20, help="Batches timed after the first batch.")
	parser.add_argument("--prior-type", default="nanotabicl")
	parser.add_argument("--prior-device", default="cpu")
	parser.add_argument("--min-features", type=int, default=2)
	parser.add_argument("--max-features", type=int, default=100)
	parser.add_argument("--max-classes", type=int, default=10)
	parser.add_argument("--min-seq-len", type=int, default=None)
	parser.add_argument("--max-seq-len", type=int, default=1024)
	parser.add_argument("--min-train-size", type=float, default=0.1)
	parser.add_argument("--max-train-size", type=float, default=0.6)
	parser.add_argument("--batch-size-per-gp", type=int, default=8)
	parser.add_argument("--batch-size-per-subgp", type=int, default=8)
	parser.add_argument("--num-threads-per-generate", type=int, default=1)
	parser.add_argument("--normalization", choices=("none", "std", "robust"), default="std")
	parser.add_argument("--icl-backend", choices=sorted(GRAPH_BACKENDS | {"encoder"}), default="graph")
	parser.add_argument("--graph-num-graphs", type=int, default=6)
	parser.add_argument("--graph-min-train-neighbors", type=int, default=4)
	parser.add_argument("--graph-max-train-neighbors", type=int, default=4)
	parser.add_argument("--graph-cross-label-fraction", type=float, default=0.25)
	parser.add_argument("--graph-train-neighbors-per-test", type=int, default=2)
	parser.add_argument("--graph-v1-prob", type=float, default=0.4)
	parser.add_argument("--graph-v2-prob", type=float, default=0.4)
	parser.add_argument("--graph-prob", type=float, default=0.2)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--pin-memory", action="store_true")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--csv", type=Path, default=None, help="Optional output CSV path.")
	args = parser.parse_args()

	if args.num_batches <= 0 or args.prefetch_factor <= 0:
		parser.error("num-batches and prefetch-factor must be positive")
	if args.batch_size_per_gp <= 0 or args.batch_size_per_subgp <= 0:
		parser.error("batch-size-per-gp and batch-size-per-subgp must be positive")
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	results = []
	for num_workers, batch_size in itertools.product(args.num_workers, args.batch_sizes):
		print(f"num_workers={num_workers} batch_size={batch_size} ...", flush=True)
		result = benchmark_case(args, num_workers, batch_size)
		results.append(result)
		print(
			f"  first_batch={result.first_batch_seconds:.3f}s "
			f"mean_batch={result.mean_batch_seconds:.3f}s "
			f"samples_per_second={result.samples_per_second:.1f}",
			flush=True,
		)

	if args.csv is not None:
		args.csv.parent.mkdir(parents=True, exist_ok=True)
		with args.csv.open("w", newline="") as file:
			writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
			writer.writeheader()
			writer.writerows(asdict(result) for result in results)
		print(f"wrote={args.csv}")


if __name__ == "__main__":
	main()
