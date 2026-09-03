from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tabicl._model.tabicl import TabICL
from tabicl._model.nanotabicl import NanoTabICLv2
from tabicl.evaluate._eval_config import build_parser
from tabicl.prior._dataset import PriorDataset


try:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
except ImportError:  # pragma: no cover - handled at runtime
    balanced_accuracy_score = None
    confusion_matrix = None


def _project_to_2d(x: np.ndarray, seed: int) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    if x.shape[1] <= 1:
        return np.concatenate([x.astype(np.float32), np.zeros((x.shape[0], 1), dtype=np.float32)], axis=1)
    if x.shape[1] == 2:
        return x.astype(np.float32)

    try:
        import umap.umap_ as umap_module

        n_neighbors = min(15, x.shape[0] - 1)
        reducer = umap_module.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        )
        return reducer.fit_transform(x).astype(np.float32)
    except ImportError:
        return x[:, :2].astype(np.float32)


@dataclass
class EvalSample:
    x: torch.Tensor
    y_full: torch.Tensor
    y_pred_test: torch.Tensor
    y_logits_test: torch.Tensor
    y_true_test: torch.Tensor
    repr_full: torch.Tensor
    train_size: int


def _build_model_from_checkpoint(ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing state_dict")

    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint is missing config")

    model_type = str(config.get("model_type", "tabicl")).lower()
    model_kwargs = {k: v for k, v in config.items() if k != "model_type"}

    if model_type == "tabicl":
        model = TabICL(**model_kwargs)
    elif model_type == "nanotabicl":
        model = NanoTabICLv2(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, model_type


def _prepare_single_dataset(batch: tuple[torch.Tensor, ...], idx: int):
    if len(batch) == 6:
        x, y, d, seq_lens, train_sizes, graph_sets = batch
    else:
        x, y, d, seq_lens, train_sizes = batch
        graph_sets = [None] * len(train_sizes)

    seq_len = int(seq_lens[idx].item())
    train_size = int(train_sizes[idx].item())
    feat_count = int(d[idx].item())

    x_i = x[idx : idx + 1, :seq_len, :feat_count].clone()
    y_i = y[idx : idx + 1, :seq_len].clone().long()
    d_i = d[idx : idx + 1].clone()
    y_train = y_i[:, :train_size]
    y_true_test = y_i[:, train_size:]

    return x_i, y_i, d_i, y_train, y_true_test, train_size, graph_sets[idx]


def _forward_with_full_repr_tabicl(model: TabICL, x_i, y_train, d_i, graph_set):
    train_size = y_train.shape[1]

    d_eff = d_i
    if d_eff is not None and len(d_eff.unique()) == 1 and int(d_eff[0].item()) == x_i.shape[-1]:
        d_eff = None
    if model.col_embedder.feature_group:
        d_eff = None

    col_embeddings = model.col_embedder(x_i, y_train=y_train, d=d_eff, embed_with_test=False)

    if model.icl_backend == "encoder":
        icl_input = model.row_interactor(col_embeddings, d=d_eff)
    elif model.icl_backend in ("graph-1d", "graph-1d-pyg"):
        icl_input = model.row_interactor(col_embeddings)
    else:
        pre_col_embeddings = model.col_embedder.project_input(x_i, d=d_eff)
        # Some feature-embedding paths keep the projected input on the source
        # device. Align it with the learned column embeddings before combining
        # them, which is required for multi-GPU evaluation such as cuda:7.
        pre_col_embeddings = pre_col_embeddings.to(device=col_embeddings.device)
        icl_input = model.icl_predictor.prepare_graph_input(
            col_embeddings=col_embeddings,
            y_train=y_train,
            pre_col_embeddings=pre_col_embeddings,
        )

    out_full, repr_full = model.icl_predictor._icl_predictions(
        icl_input,
        y_train,
        return_pre_decoder_repr=True,
        graph_set=graph_set,
    )

    pred_test = out_full[:, train_size:]
    # Prior labels may use arbitrary global class IDs (NanoTabICL remaps
    # classes, e.g. binary labels can be [2, 7]). Do not truncate logits to
    # [:num_classes], because that assumes the active IDs are [0, ..., K-1]
    # and can discard the logits for the actual target classes. Keeping the
    # complete output lets argmax return the same global IDs as y_test.

    return pred_test, repr_full


def _forward_with_full_repr_nano(model: NanoTabICLv2, x_i, y_train):
    pred_test, repr_test = model(x_i, y_train, return_pre_decoder_repr=True)

    # NanoTabICLv2 returns only test representations; no explicit train representations are available.
    train_size = y_train.shape[1]
    zeros_train = torch.zeros(
        (repr_test.shape[0], train_size, repr_test.shape[-1]),
        dtype=repr_test.dtype,
        device=repr_test.device,
    )
    repr_full = torch.cat([zeros_train, repr_test], dim=1)
    return pred_test, repr_full


def _evaluate_one_dataset(model, model_type: str, batch: tuple[torch.Tensor, ...], idx: int, device: str) -> EvalSample:
    x_i, y_i, d_i, y_train, y_true_test, train_size, graph_set = _prepare_single_dataset(batch, idx)

    x_i = x_i.to(device)
    y_i = y_i.to(device)
    d_i = d_i.to(device)
    y_train = y_train.to(device)
    y_true_test = y_true_test.to(device)

    with torch.no_grad():
        if model_type == "tabicl":
            pred_test, repr_full = _forward_with_full_repr_tabicl(model, x_i, y_train, d_i, graph_set)
        else:
            pred_test, repr_full = _forward_with_full_repr_nano(model, x_i, y_train)

    y_pred_test = pred_test.argmax(dim=-1)

    return EvalSample(
        x=x_i.detach().cpu().squeeze(0),
        y_full=y_i.detach().cpu().squeeze(0),
        y_pred_test=y_pred_test.detach().cpu().squeeze(0),
        y_logits_test=pred_test.detach().cpu().squeeze(0),
        y_true_test=y_true_test.detach().cpu().squeeze(0),
        repr_full=repr_full.detach().cpu().squeeze(0),
        train_size=train_size,
    )


def _build_figure(sample: EvalSample, cm: np.ndarray, output_path: Path, seed: int):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ModuleNotFoundError("matplotlib is required for figure output") from exc

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    x_np = sample.x.numpy()
    y_np = sample.y_full.numpy().astype(int)
    train_size = sample.train_size

    y_train = y_np[:train_size]
    y_test_pred = sample.y_pred_test.numpy().astype(int)

    x_emb = _project_to_2d(x_np, seed=seed)
    ax = axes[0]
    ax.scatter(x_emb[:train_size, 0], x_emb[:train_size, 1], c=y_train, marker="o", s=20, alpha=0.9)
    ax.scatter(x_emb[train_size:, 0], x_emb[train_size:, 1], c=y_test_pred, marker="x", s=25, alpha=0.9)
    ax.set_title("Raw Data Embedding")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.grid(alpha=0.2)

    repr_np = sample.repr_full.numpy()
    r_emb = _project_to_2d(repr_np, seed=seed + 1)
    ax = axes[1]
    ax.scatter(r_emb[:train_size, 0], r_emb[:train_size, 1], c=y_train, marker="o", s=20, alpha=0.9)
    ax.scatter(r_emb[train_size:, 0], r_emb[train_size:, 1], c=y_test_pred, marker="x", s=25, alpha=0.9)
    ax.set_title("Pre-Decoder Representation UMAP")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)

    ax = axes[2]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_eval(args):
    if balanced_accuracy_score is None or confusion_matrix is None:
        raise ModuleNotFoundError("scikit-learn is required for evaluation metrics")

    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)

    model, model_type = _build_model_from_checkpoint(args.checkpoint_path, args.device)

    prior_dataset = PriorDataset(
        batch_size=args.eval_batch_size,
        batch_size_per_gp=args.batch_size_per_gp,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        log_seq_len=args.log_seq_len,
        seq_len_per_gp=args.seq_len_per_gp,
        min_train_size=args.min_train_size,
        max_train_size=args.max_train_size,
        replay_small=args.replay_small,
        prior_type=args.prior_type,
        device=args.prior_device,
        n_jobs=1,
    )
    prior_loader = DataLoader(prior_dataset, batch_size=None, shuffle=False, num_workers=0)

    scores: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    selected_sample: EvalSample | None = None

    processed = 0
    prior_iter = iter(prior_loader)

    while processed < args.num_datasets:
        batch = next(prior_iter)
        fields = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
        bsz = int(fields[0].shape[0])

        for idx in range(bsz):
            if processed >= args.num_datasets:
                break

            sample = _evaluate_one_dataset(model, model_type, tuple(fields), idx, args.device)

            y_true_np = sample.y_true_test.numpy().astype(int)
            y_pred_np = sample.y_pred_test.numpy().astype(int)
            score = float(balanced_accuracy_score(y_true_np, y_pred_np))

            scores.append(score)
            y_true_all.append(y_true_np)
            y_pred_all.append(y_pred_np)

            if selected_sample is None:
                selected_sample = sample

            processed += 1

    if not scores:
        raise RuntimeError("No datasets were evaluated")

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)
    labels = np.unique(np.concatenate([y_true_concat, y_pred_concat]))
    cm = confusion_matrix(y_true_concat, y_pred_concat, labels=labels)

    if selected_sample is None:
        raise RuntimeError("Could not build a visualization sample")

    _build_figure(
        sample=selected_sample,
        cm=cm,
        output_path=Path(args.output_figure_path),
        seed=args.np_seed,
    )

    mean_bacc = float(np.mean(scores))
    std_bacc = float(np.std(scores))

    print(f"Evaluated datasets: {len(scores)}")
    print(f"Balanced accuracy mean: {mean_bacc:.6f}")
    print(f"Balanced accuracy std: {std_bacc:.6f}")
    print(f"Saved figure: {args.output_figure_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_eval(args)
