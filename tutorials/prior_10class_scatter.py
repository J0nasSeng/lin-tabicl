"""
Visualize a 10-class dataset sampled from the TabICL prior.

Usage examples
--------------
python tutorials/prior_10class_scatter.py --n-features 2
python tutorials/prior_10class_scatter.py --n-features 16 --seq-len 512
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from tabicl.prior import PriorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and visualize a 10-class prior dataset.")
    parser.add_argument("--n-features", type=int, default=2, help="Number of features N in the sampled dataset.")
    parser.add_argument("--seq-len", type=int, default=1024, help="Number of samples T in the sampled dataset.")
    parser.add_argument(
        "--prior-type",
        type=str,
        default="nanotabicl",
        choices=["mlp_scm", "tree_scm", "mix_scm", "nanotabicl"],
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used by prior generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="Maximum attempts to sample a dataset with exactly 10 classes.",
    )
    return parser.parse_args()


def build_prior(n_features: int, seq_len: int, prior_type: str, device: str) -> PriorDataset:
    return PriorDataset(
        batch_size=1,
        batch_size_per_gp=1,
        min_features=n_features,
        max_features=n_features,
        max_classes=10,
        min_seq_len=seq_len,
        max_seq_len=2*seq_len,
        min_train_size=0.2,
        max_train_size=0.5,
        prior_type=prior_type,
        n_jobs=1,
        device=device,
    )


def sample_exact_10_class_dataset(prior: PriorDataset, max_attempts: int) -> tuple[np.ndarray, np.ndarray]:
    for attempt in range(1, max_attempts + 1):
        X, y, d, _, _ = prior.get_batch()
        x0 = X[0, :, : int(d[0].item())]
        y0 = y[0].long()

        print(y0.numel(), torch.unique(y0).numel())

        if torch.unique(y0).numel() == 10:
            print(f"Found a 10-class dataset after {attempt} attempt(s).")
            return x0.detach().cpu().numpy(), y0.detach().cpu().numpy()

    raise RuntimeError(
        f"Could not sample a dataset with exactly 10 classes in {max_attempts} attempts. "
        "Try increasing --max-attempts or --seq-len."
    )


def to_2d(X: np.ndarray, seed: int) -> np.ndarray:
    if X.shape[1] == 2:
        return X

    try:
        import umap.umap_ as umap
    except ImportError as exc:
        raise ImportError(
            "UMAP is required when n_features > 2. Install with: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(X)


def plot_colored_scatter(X2: np.ndarray, y: np.ndarray, n_features: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="tab10", s=14, alpha=0.9)

    title = "Prior dataset (10 classes)"
    if n_features == 2:
        title += " - direct 2D"
    else:
        title += f" - UMAP from {n_features}D"

    ax.set_title(title)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Class label")
    cbar.set_ticks(range(10))

    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()

    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)

    prior = build_prior(
        n_features=args.n_features,
        seq_len=args.seq_len,
        prior_type=args.prior_type,
        device=args.device,
    )

    X, y = sample_exact_10_class_dataset(prior=prior, max_attempts=args.max_attempts)
    X2 = to_2d(X, seed=args.seed)
    plot_colored_scatter(X2, y, n_features=args.n_features)


if __name__ == "__main__":
    main()
