from __future__ import annotations

import os
import timeit
import warnings
import functools
from contextlib import nullcontext

import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
except ImportError:
    RandomForestClassifier = None
    balanced_accuracy_score = None

from tabicl._model.tabicl import TabICL
from tabicl._model.nanotabicl import NanoTabICLv2


GRAPH_BACKENDS = {"graph", "graph-pyg", "graph-2d", "graph-2d-pyg", "graph-1d", "graph-1d-pyg"}
from tabicl.prior._dataset import PriorDataset
from tabicl.prior._genload import LoadPriorDataset
from tabicl.train._optim import get_scheduler
from tabicl.train._losses import entropy_regularizer, supervised_contrastive_loss
from tabicl.train._train_config import build_parser
from tabicl.train._umap_logging import build_test_umap_wandb_images
from rtpt import RTPT

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


def build_confusion_matrix_plot_image(cm: list[list[int]], class_names: list[str], wandb_module):
    """Render confusion matrix heatmap and wrap it as a W&B image artifact."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (Sample)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    threshold = cm_arr.max() / 2.0 if cm_arr.size > 0 else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = int(cm_arr[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    image = wandb_module.Image(fig)
    plt.close(fig)
    return image


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper

class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config):
        self.config = config
        self.train_col_embed_only = bool(getattr(config, "train_col_embed_only", False))
        if self.train_col_embed_only:
            self._validate_column_embedding_only_mode()
        self.configure_ddp()
        if self.config.batch_size <= 0 or self.config.micro_batch_size <= 0:
            raise ValueError("batch_size and micro_batch_size must be positive")
        if self.config.batch_size % self.config.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        if self.config.batch_size_per_gp != self.config.micro_batch_size:
            raise ValueError("batch_size_per_gp must equal micro_batch_size for homogeneous feature batches")
        self.accumulation_steps = self.config.batch_size // self.config.micro_batch_size
        self.build_model()
        self.configure_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()
        self.configure_wandb()
        self.rtpt = RTPT(name_initials="JS", experiment_name="TabICL_Stage1", max_iterations=self.config.max_steps)
        self.rtpt.start()

    def _validate_column_embedding_only_mode(self):
        """Validate the opt-in stage 1.5 checkpoint-transfer mode."""
        if str(getattr(self.config, "model_type", "tabicl")).lower() != "tabicl":
            raise ValueError("--train-col-embed-only is supported only for model_type='tabicl'.")
        if not getattr(self.config, "checkpoint_path", None):
            raise ValueError("--train-col-embed-only requires an explicit --checkpoint_path.")
        if not os.path.isfile(self.config.checkpoint_path):
            raise FileNotFoundError(
                f"--train-col-embed-only checkpoint not found: {self.config.checkpoint_path}"
            )
        if getattr(self.config, "freeze_col", False):
            raise ValueError("--train-col-embed-only cannot be combined with --freeze_col=True.")
        if self.config.max_features <= 0:
            raise ValueError("--max_features must be positive.")

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        if self.master_process:
            print(
                "Runtime device setup: "
                f"device={self.config.device}, "
                f"ddp={self.ddp}, rank={self.ddp_rank}, local_rank={self.ddp_local_rank}, world_size={self.ddp_world_size}"
            )

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""

        if self.config.wandb_log and self.master_process:
            if wandb is None:
                raise ModuleNotFoundError(
                    "wandb is not installed but --wandb_log=True. Install with `uv sync --extra pretrain` "
                    "or run with --wandb_log False."
                )
            if self.config.checkpoint_dir:
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            if self.config.wandb_dir:
                os.makedirs(self.config.wandb_dir, exist_ok=True)

            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            checkpoint_exists = False
            if getattr(self.config, "checkpoint_path", None):
                checkpoint_exists = os.path.exists(self.config.checkpoint_path)
            elif getattr(self.config, "checkpoint_dir", None):
                checkpoint_exists = self.get_latest_checkpoint() is not None

            # Reuse a stored run id only when a checkpoint exists to resume from.
            if self.config.wandb_id is None and checkpoint_exists and os.path.exists(id_path):
                with open(id_path, "r") as f:
                    self.config.wandb_id = f.read().strip()

            resume_mode = "allow" if self.config.wandb_id is not None else "never"

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume=resume_mode,
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        """Build and initialize the selected training model."""

        model_type = str(getattr(self.config, "model_type", "tabicl")).lower()
        self.model_type = model_type

        self.model_config = {
            "model_type": model_type,
            "max_classes": self.config.max_classes,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "icl_backend": self.config.icl_backend,
            "icl_decoder_type": self.config.icl_decoder_type,
            "icl_soft_kmeans_temperature": self.config.icl_soft_kmeans_temperature,
            "graph_min_train_neighbors": self.config.graph_min_train_neighbors,
            "graph_max_train_neighbors": self.config.graph_max_train_neighbors,
            "graph_cross_label_fraction": self.config.graph_cross_label_fraction,
            "graph_train_neighbors_per_test": self.config.graph_train_neighbors_per_test,
            "graph_seed": self.config.graph_seed,
            "graph_share_across_batch": self.config.graph_share_across_batch,
            "graph_num_graphs": getattr(self.config, "graph_num_graphs", None) or self.config.icl_num_blocks,
            "learnable_residual": getattr(self.config, "learnable_residual", False),
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
            "recompute": getattr(self.config, "recompute", False),
        }

        if model_type == "tabicl":
            # Persist this field for new full TabICL checkpoints. Historical
            # checkpoints may omit it and are handled by shape inference when
            # the stage 1.5 loader reads their identity table.
            self.model_config["max_features"] = self.config.max_features
            model = TabICL(**{k: v for k, v in self.model_config.items() if k != "model_type"})
        elif model_type == "nanotabicl":
            if self.config.max_classes <= 0:
                raise ValueError("NanoTabICL training currently requires max_classes > 0 in this Trainer.")

            self.nano_model_config = {
                "model_type": model_type,
                "max_classes": self.config.max_classes,
                "out_dim": self.config.max_classes,
                "embed_dim": self.config.embed_dim,
                "col_num_blocks": self.config.col_num_blocks,
                "row_num_blocks": self.config.row_num_blocks,
                "icl_num_blocks": self.config.icl_num_blocks,
                "col_nhead": self.config.col_nhead,
                "row_nhead": self.config.row_nhead,
                "icl_nhead": self.config.icl_nhead,
                "n_cls_cols": self.config.row_num_cls,
                "n_cls_rows": self.config.col_num_inds,
            }
            model = NanoTabICLv2(**{k: v for k, v in self.nano_model_config.items() if k != "model_type"})
            self.model_config = self.nano_model_config
        else:
            raise ValueError(f"Unknown model_type={model_type}. Expected 'tabicl' or 'nanotabicl'.")

        model.to(device=self.config.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        # Freeze model components if requested. Stage 1.5 deliberately keeps
        # the complete column embedder trainable and freezes everything else.
        if self.train_col_embed_only:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in model.col_embedder.parameters():
                parameter.requires_grad_(True)
            model.train()
            model.col_embedder.train()
            model.row_interactor.eval()
            model.icl_predictor.eval()
        elif model_type == "tabicl" and self.config.freeze_col:
            model.col_embedder.eval()
            for param in model.col_embedder.parameters():
                param.requires_grad = False

        if model_type == "tabicl" and self.config.freeze_row:
            model.row_interactor.eval()
            for param in model.row_interactor.parameters():
                param.requires_grad = False

        if model_type == "tabicl" and self.config.freeze_icl:
            model.icl_predictor.eval()
            for param in model.icl_predictor.parameters():
                param.requires_grad = False

        if model_type == "nanotabicl" and (self.config.freeze_col or self.config.freeze_row or self.config.freeze_icl):
            raise ValueError("freeze_col/freeze_row/freeze_icl are only supported for model_type='tabicl'.")

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            # Backend-specific execution paths can leave whole parameter groups
            # unused in a given step (e.g., graph backend bypassing row/encoder stacks).
            self.model = DDP(
                model,
                device_ids=[self.ddp_local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def _build_prior_dataset(self, *, is_validation: bool):
        """Build train/validation prior dataset instances with separated streams."""

        if self.config.prior_dir is None:
            return PriorDataset(
                batch_size=self.config.micro_batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                device=self.config.prior_device,
                n_jobs=1,  # Set to 1 to avoid nested parallelism during DDP
                normalization=getattr(self.config, "normalization", "none"),
                graph_backend=getattr(self.config, "icl_backend", None) in GRAPH_BACKENDS,
                graph_num_graphs=getattr(self.config, "graph_num_graphs", None) or self.config.icl_num_blocks,
                graph_min_train_neighbors=self.config.graph_min_train_neighbors,
                graph_max_train_neighbors=self.config.graph_max_train_neighbors,
                graph_cross_label_fraction=self.config.graph_cross_label_fraction,
                graph_train_neighbors_per_test=self.config.graph_train_neighbors_per_test,
                graph_seed=self.config.graph_seed,
                graph_share_across_batch=self.config.graph_share_across_batch,
                graph_v1_prob=getattr(self.config, "graph_v1_prob", 1.0),
                graph_v2_prob=getattr(self.config, "graph_v2_prob", 0.0),
                graph_prob=getattr(self.config, "graph_prob", 0.0),
            )

        val_start_offset = max(1, int(self.config.max_steps))
        start_from = self.config.load_prior_start + (val_start_offset if is_validation else 0)
        return LoadPriorDataset(
            data_dir=self.config.prior_dir,
            batch_size=self.config.micro_batch_size,
            ddp_world_size=self.ddp_world_size,
            ddp_rank=self.ddp_rank,
            start_from=start_from,
            delete_after_load=(self.config.delete_after_load if not is_validation else False),
            device=self.config.prior_device,
            normalization=getattr(self.config, "normalization", "none"),
            graph_backend=getattr(self.config, "icl_backend", None) in GRAPH_BACKENDS,
        )

    def _build_prior_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=None,  # No additional batching since prior dataset handles batching internally
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=self.config.device if self.config.prior_device == "cpu" else "",
            persistent_workers=True,
        )

    def configure_prior(self):
        """Set up tabular train/validation prior data generators and dataloaders."""

        train_dataset = self._build_prior_dataset(is_validation=False)
        val_dataset = self._build_prior_dataset(is_validation=True)

        if self.master_process:
            print(train_dataset)

        self.train_dataloader = self._build_prior_dataloader(train_dataset)
        self.val_dataloader = self._build_prior_dataloader(val_dataset)

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""

        trainable_parameters = [parameter for parameter in self.raw_model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters remain after applying freeze settings.")
        self.optimizer = optim.AdamW(
            params=trainable_parameters, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.config.dtype not in dtype_map:
            raise ValueError(f"Unsupported training dtype: {self.config.dtype}")

        autocast_dtype = dtype_map[self.config.dtype]
        use_grad_scaler = self.amp and self.config.dtype == "float16"
        self.scaler = torch.GradScaler("cuda", enabled=use_grad_scaler)

        if self.amp:
            if self.master_process:
                print(
                    f"Automatic Mixed Precision is enabled. "
                    f"compute_dtype={self.config.dtype}, grad_scaler={use_grad_scaler}"
                )
            self.amp_ctx = torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
            )
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        if self.train_col_embed_only:
            self._load_expanded_column_checkpoint(checkpoint)
        else:
            self.raw_model.load_state_dict(checkpoint["state_dict"])

        # Optionally load optimizer and scheduler state
        if self.config.only_load_model or self.train_col_embed_only:
            print("Only loading model weights")
        elif not self.train_col_embed_only:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")

    def _load_expanded_column_checkpoint(self, checkpoint):
        """Load a checkpoint while expanding its column identity table."""
        source_config = checkpoint.get("config") or {}
        if str(source_config.get("model_type", "tabicl")).lower() != "tabicl":
            raise ValueError("--train-col-embed-only requires a TabICL checkpoint.")

        source_state = checkpoint["state_dict"]
        identity_key = "col_embedder.column_identity_rotations"
        source_identity = source_state.get(identity_key)
        target_state = self.raw_model.state_dict()
        target_identity = target_state.get(identity_key)
        if source_identity is None or target_identity is None:
            raise ValueError("Checkpoint/model does not contain column identity embeddings.")
        if source_identity.ndim != target_identity.ndim or source_identity.shape[1:] != target_identity.shape[1:]:
            raise ValueError(
                "Column identity embedding dimensions are incompatible: "
                f"checkpoint={tuple(source_identity.shape)}, model={tuple(target_identity.shape)}"
            )
        source_max_features = int(source_identity.shape[0])
        configured_source_max_features = source_config.get("max_features")
        if configured_source_max_features is not None and int(configured_source_max_features) != source_max_features:
            raise ValueError(
                "Checkpoint max_features does not match its column identity embedding shape: "
                f"config={configured_source_max_features}, tensor={source_max_features}"
            )
        if source_max_features > self.config.max_features:
            raise ValueError(
                f"Target max_features={self.config.max_features} is smaller than checkpoint capacity "
                f"{source_max_features}."
            )

        missing = set(target_state) - set(source_state) - {identity_key}
        unexpected = set(source_state) - set(target_state)
        mismatched = {
            key: (tuple(source_state[key].shape), tuple(target_state[key].shape))
            for key in set(source_state) & set(target_state)
            if key != identity_key and source_state[key].shape != target_state[key].shape
        }
        if missing or unexpected or mismatched:
            raise ValueError(
                "Checkpoint architecture is incompatible with the stage 1.5 model: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, mismatched={mismatched}"
            )

        expanded_identity = target_identity.clone()
        expanded_identity[:source_max_features].copy_(source_identity.to(expanded_identity))
        target_state[identity_key] = expanded_identity
        self.raw_model.load_state_dict(target_state, strict=True)
        print(
            f"Loaded checkpoint column embeddings: copied {source_max_features} rows, "
            f"initialized {self.config.max_features - source_max_features} new rows"
        )

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """Manage temporary checkpoints by deleting the oldest when limit is exceeded."""
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """

        if self.master_process:
            step_progress = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        train_dataloader = iter(self.train_dataloader)
        val_dataloader = iter(self.val_dataloader)
        for step in step_progress:
            # Consume a logical batch as a stream of prefetched micro-batches.
            with Timer() as train_timer:
                results, prior_time, train_dataloader = self.run_batch(train_dataloader)

                if self._should_log_conf_mat_now():
                    val_results, val_prior_time, val_dataloader = self.run_validation_logging_batch(val_dataloader)
                    results.update(val_results)
                    results["val_prior_time"] = val_prior_time
            train_time = train_timer.elapsed

            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()

            self.curr_step = step + 1
            if self.master_process:
                # Add timing information to results
                results.update({"prior_time": prior_time, "train_time": train_time})

                # Update progress bar with rounded values for cleaner display
                display_results = {}
                for k, v in results.items():
                    if isinstance(v, (int, float, np.floating)):
                        display_results[k] = round(float(v), 3)
                if display_results:
                    step_progress.set_postfix(**display_results)

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    # Manage checkpoint limit only for temporary checkpoints
                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                # Add learning rate to results
                results["lr"] = self.scheduler.get_last_lr()[0]
                results.update(self._get_gat_alpha_metrics())
                wandb_step = int(self.curr_step)
                run_step = getattr(self.wandb_run, "step", None)
                if isinstance(run_step, int):
                    wandb_step = max(wandb_step, run_step)
                wandb.log(results, step=wandb_step)
            
            self.rtpt.step()

    def _should_log_conf_mat_now(self) -> bool:
        log_every = int(getattr(self.config, "log_conf_mat_every", 100))
        return log_every > 0 and ((self.curr_step + 1) % log_every == 0)

    def _get_gat_alpha_metrics(self) -> dict[str, float]:
        """Return effective residual strengths for all GAT layers.

        ``GraphMultiheadAttention.alpha`` is stored as an unconstrained logit;
        the forward pass uses ``sigmoid(alpha)``. Log the effective coefficient
        used by the network rather than the underlying parameter so the W&B
        plots directly show the residual/attention mixing strength.
        """
        if getattr(self.raw_model, "icl_backend", None) != "graph":
            return {}

        predictor = getattr(self.raw_model, "icl_predictor", None)
        if not getattr(predictor, "learnable_residual", False):
            return {}
        graph_blocks = getattr(predictor, "gat_icl", None)
        graph_blocks = getattr(graph_blocks, "graph_blocks", ())
        return {
            f"gat/graph_block_{layer_idx}/alpha": float(torch.sigmoid(block.attn.alpha).detach().cpu())
            for layer_idx, block in enumerate(graph_blocks)
            if hasattr(getattr(block, "attn", None), "alpha")
        }

    def _prepare_padded_batch(self, batch):
        return [
            t.to_padded_tensor(padding=0.0) if hasattr(t, "is_nested") and t.is_nested else t
            for t in batch
        ]

    def _prepare_micro_batch_tensors(self, micro_batch):
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch[:5]
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.device, non_blocking=True)
        micro_y = micro_y.to(self.config.device, non_blocking=True)
        micro_d = micro_d.to(self.config.device, non_blocking=True)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]
        graph_sets = micro_batch[5] if len(micro_batch) == 6 else None
        return micro_X, micro_y, micro_d, y_train, y_test, graph_sets

    def _build_logging_payload(
        self,
        *,
        pred_3d,
        y_test,
        true,
        pre_decoder_repr_test,
        dataset_results_to_log,
        dataset_results_start_index,
    ):
        payload = {}
        if confusion_matrix is not None and true.numel() > 0:
            payload["confusion_matrix_sample_y_true"] = true.detach().cpu().tolist()
            payload["confusion_matrix_sample_y_pred"] = pred_3d.flatten(end_dim=-2).argmax(dim=1).detach().cpu().tolist()

            if (
                dataset_results_to_log > 0
                and self.master_process
                and self.wandb_run is not None
                and wandb is not None
            ):
                class_names = [str(i) for i in range(self.config.max_classes)]
                confusion_images = {}
                n_ds = min(dataset_results_to_log, y_test.shape[0], pred_3d.shape[0])
                y_pred_per_ds = pred_3d.argmax(dim=-1)
                labels = np.arange(self.config.max_classes)

                for local_idx in range(n_ds):
                    y_true_ds = y_test[local_idx].long().detach().cpu().numpy()
                    y_pred_ds = y_pred_per_ds[local_idx].long().detach().cpu().numpy()
                    cm_ds = confusion_matrix(y_true_ds, y_pred_ds, labels=labels)
                    cm_image = build_confusion_matrix_plot_image(
                        cm=cm_ds.tolist(),
                        class_names=class_names,
                        wandb_module=wandb,
                    )
                    if cm_image is not None:
                        key = f"confusion_matrix_ds_{dataset_results_start_index + local_idx}"
                        confusion_images[key] = cm_image

                if confusion_images:
                    payload["confusion_images"] = confusion_images

        if (
            dataset_results_to_log > 0
            and self.master_process
            and self.wandb_run is not None
            and pre_decoder_repr_test is not None
        ):
            umap_images = build_test_umap_wandb_images(
                repr_test=pre_decoder_repr_test,
                y_test=y_test,
                wandb_module=wandb,
                max_datasets=dataset_results_to_log,
                start_index=dataset_results_start_index,
                seed=self.config.np_seed + self.curr_step,
            )
            if umap_images:
                payload["umap_images"] = umap_images

        return payload

    def _merge_micro_results(self, results, micro_results, payload_state):
        for k, v in micro_results.items():
            if k == "confusion_matrix_sample_y_true":
                payload_state["confusion_y_true_payload"].extend(v)
                continue
            if k == "confusion_matrix_sample_y_pred":
                payload_state["confusion_y_pred_payload"].extend(v)
                continue
            if k == "confusion_images":
                payload_state["confusion_plot_payload"].update(v)
                payload_state["dataset_results_next_index"] = max(
                    len(payload_state["confusion_plot_payload"]),
                    len(payload_state["umap_payload"]),
                )
                payload_state["dataset_results_remaining"] = max(0, 4 - payload_state["dataset_results_next_index"])
                continue
            if k == "umap_images":
                payload_state["umap_payload"].update(v)
                payload_state["dataset_results_next_index"] = max(
                    len(payload_state["confusion_plot_payload"]),
                    len(payload_state["umap_payload"]),
                )
                payload_state["dataset_results_remaining"] = max(0, 4 - payload_state["dataset_results_next_index"])
                continue

            if k not in results:
                results[k] = 0.0
            results[k] += v

    @staticmethod
    def _finalize_metric_results(results, divisor: int):
        """Average accumulated tensor metrics and materialize Python scalars once."""
        for key, value in list(results.items()):
            if isinstance(value, torch.Tensor):
                results[key] = (value / divisor).item()
        return results

    def _finalize_confusion_payload(self, payload_state):
        payload = {}
        y_true_payload = payload_state["confusion_y_true_payload"]
        y_pred_payload = payload_state["confusion_y_pred_payload"]
        if (
            confusion_matrix is not None
            and len(y_true_payload) > 0
            and len(y_pred_payload) > 0
        ):
            labels = np.arange(self.config.max_classes)
            cm = confusion_matrix(y_true_payload, y_pred_payload, labels=labels)
            confusion_matrix_payload = cm.tolist()
            payload["confusion_matrix_sample"] = confusion_matrix_payload
            if self.wandb_run is not None and wandb is not None:
                class_names = [str(i) for i in range(self.config.max_classes)]
                cm_image = build_confusion_matrix_plot_image(
                    cm=confusion_matrix_payload,
                    class_names=class_names,
                    wandb_module=wandb,
                )
                if cm_image is not None:
                    payload["confusion_matrix_sample_plot"] = cm_image

        if payload_state["confusion_plot_payload"]:
            payload.update(payload_state["confusion_plot_payload"])
        if payload_state["umap_payload"]:
            payload.update(payload_state["umap_payload"])
        return payload

    def _apply_optimizer_updates(self):
        # Always unscale once before any grad inspection/clipping/step so logged norms
        # reflect true (unscaled) gradients when AMP GradScaler is enabled.
        self.scaler.unscale_(self.optimizer)

        grad_sq_sum = None
        for param in self.model.parameters():
            if param.grad is None:
                continue
            param_grad_sq_sum = param.grad.detach().float().pow(2).sum()
            grad_sq_sum = param_grad_sq_sum if grad_sq_sum is None else grad_sq_sum + param_grad_sq_sum
        global_grad_norm = grad_sq_sum.sqrt().item() if grad_sq_sum is not None else 0.0

        if self.config.gradient_clipping > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return global_grad_norm

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """Validate consistent sequence length and train size within a micro batch.

        Ensures all datasets in a micro batch share the same sequence length and
        train/test split position, required for efficient batch processing during
        gradient accumulation.

        Parameters
        ----------
        micro_seq_len : Tensor
            Sequence lengths for each dataset, shape ``(micro_batch_size,)``.

        micro_train_size : Tensor
            Training sizes (split positions) for each dataset, shape
            ``(micro_batch_size,)``.

        Returns
        -------
        seq_len : int
            The common sequence length for the micro batch.

        train_size : int
            The common train size for the micro batch.

        Raises
        ------
        ValueError
            If sequence lengths or train sizes are inconsistent.
        """
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError("All datasets in the micro batch must have the same sequence length.")

        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError("All datasets in the micro batch must have the same training size.")

        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()

        return seq_len, train_size

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        """Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (``micro_d.max()``) respectively. This
        optimizes memory and computation by removing unused tensor elements.

        Parameters
        ----------
        micro_X : Tensor
            Input features per dataset of shape ``(B, T, H)``.

        micro_y : Tensor
            Target labels per dataset of shape ``(B, T)``.

        micro_d : Tensor
            Number of active features per dataset of shape ``(B,)``.

        seq_len : int
            Validated sequence length for this micro batch.

        Returns
        -------
        micro_X : Tensor
            Truncated features of shape ``(B, seq_len, micro_d.max())``.

        micro_y : Tensor
            Truncated labels of shape ``(B, seq_len)``.
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]

        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        # Truncate feature dimension
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]

        return micro_X, micro_y

    def fit_ensemble(self, micro_batch):
        """Debug helper: fit one random-forest ensemble per dataset.

        Uses only the train partition of each dataset in the micro batch and
        prints balanced accuracy on the corresponding test partition.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_train_size)
            matching the format expected by ``run_micro_batch``.
        """
        if RandomForestClassifier is None or balanced_accuracy_score is None:
            raise ModuleNotFoundError(
                "scikit-learn is required for fit_ensemble debugging. "
                "Install with `uv sync --extra pretrain` or add sklearn to your environment."
            )

        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size, _graph_sets = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        X_cpu = micro_X.detach().cpu()
        y_cpu = micro_y.detach().cpu().long()
        d_cpu = micro_d.detach().cpu().long()

        print(f"fit_ensemble debug: datasets={X_cpu.shape[0]}, train_size={train_size}, test_size={seq_len - train_size}")

        accuracies = []

        for ds_idx in range(X_cpu.shape[0]):
            d_i = int(d_cpu[ds_idx].item())
            X_ds = X_cpu[ds_idx, :, :d_i].numpy()
            y_ds = y_cpu[ds_idx].numpy()

            X_train, X_test = X_ds[:train_size], X_ds[train_size:]
            y_train, y_test = y_ds[:train_size], y_ds[train_size:]

            if X_test.shape[0] == 0:
                print(f"  dataset {ds_idx}: skipped (empty test set)")
                continue

            clf = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            bacc = balanced_accuracy_score(y_test, y_pred)
            accuracies.append(bacc)
            print(f"  dataset {ds_idx}: balanced_accuracy={bacc:.4f}")
        print(f"fit_ensemble debug: average balanced_accuracy={np.mean(accuracies):.4f}")

    def run_micro_batch(
        self,
        micro_batch,
        micro_batch_idx,
        num_micro_batches,
        collect_artifacts: bool = False,
        dataset_results_to_log: int = 2,
        dataset_results_start_index: int = 0,
        do_backward: bool = True,
        sync_gradients: bool = True,
        accumulation_divisor: int = 1,
    ):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_train_size) tensors
            for the micro batch.

        micro_batch_idx : int
            Index of the current micro batch.

        num_micro_batches : int
            Total number of micro batches.

        Returns
        -------
        dict
            Result dictionary with 'ce' and 'accuracy' keys.
        """

        # debugging
        #self.fit_ensemble(micro_batch)

        micro_X, _micro_y, micro_d, y_train, y_test, graph_sets = self._prepare_micro_batch_tensors(micro_batch)
        graph_set = graph_sets if graph_sets is not None else None

        # Set DDP gradient sync for last micro batch only
        if do_backward and self.ddp:
            self.model.require_backward_grad_sync = (
                sync_gradients and micro_batch_idx == num_micro_batches - 1
            )

        with self.amp_ctx:
            supcon_weight = float(getattr(self.config, "supcon_weight", 0.0))
            entropy_weight = float(getattr(self.config, "entropy_weight", 0.0))
            capture_pre_decoder_repr = collect_artifacts or (supcon_weight > 0)
            model_kwargs = {"return_pre_decoder_repr": capture_pre_decoder_repr}
            if getattr(self.raw_model, "icl_backend", None) in GRAPH_BACKENDS:
                if graph_set is None:
                    raise ValueError("Graph backend requires precomputed graph metadata in the prior batch")
                model_kwargs["graph_set"] = graph_set
            model_out = self.model(micro_X, y_train, micro_d, **model_kwargs)
            pre_decoder_repr_test = None
            if capture_pre_decoder_repr:
                # Some models/paths (e.g. TabICL in eval mode) may return only
                # predictions even when return_pre_decoder_repr=True.
                if isinstance(model_out, tuple):
                    if len(model_out) >= 2:
                        pred_3d = model_out[0]
                        pre_decoder_repr_test = model_out[1]
                    else:
                        pred_3d = model_out[0]
                else:
                    pred_3d = model_out
            else:
                pred_3d = model_out

            pred = pred_3d.flatten(end_dim=-2)
            true = y_test.long().flatten()
            # All datasets have the same sequence length within a micro-batch,
            # so flattening preserves the previous mean-over-datasets reduction
            # while avoiding one cross-entropy kernel and Python loop per dataset.
            ce_loss = F.cross_entropy(
                pred,
                true,
                label_smoothing=float(getattr(self.config, "label_smoothing", 0.1)),
                reduction="mean",
            )

            supcon_raw_loss = pred.new_zeros(())
            if supcon_weight > 0 and pre_decoder_repr_test is not None:
                # Compute SupCon independently for each dataset. Flattening the
                # whole micro-batch would make equal labels from different
                # datasets positive pairs, although their representations are
                # not expected to be close.
                supcon_losses = []
                for ds_idx in range(pre_decoder_repr_test.shape[0]):
                    ds_loss = supervised_contrastive_loss(
                        pre_decoder_repr_test[ds_idx],
                        y_test[ds_idx].long(),
                        temperature=0.2
                    )
                    supcon_losses.append(ds_loss)

                if supcon_losses:
                    supcon_raw_loss = torch.stack(supcon_losses).mean()

            entropy_raw_loss = pred.new_zeros(())
            if entropy_weight > 0:
                entropy_input_type = (
                    "log_probs"
                    if self.raw_model.icl_predictor.decoder_type in ("soft_kmeans", "rbf", "euclidean")
                    else "logits"
                )
                entropy_raw_loss = entropy_regularizer(pred_3d, input_type=entropy_input_type)

            # Keep CE as primary objective; maximize predictive entropy via a negative entropy term.
            loss = ce_loss + supcon_weight * supcon_raw_loss - entropy_weight * entropy_raw_loss

        if do_backward:
            scaled_loss = loss / accumulation_divisor
            self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {}
            micro_results["ce"] = ce_loss.detach()
            micro_results["loss"] = loss.detach()
            micro_results["scale"] = pred.new_tensor(self.scaler.get_scale())
            accuracy = (pred.argmax(dim=1) == true).float().mean()
            micro_results["accuracy"] = accuracy.detach()
            if supcon_weight > 0:
                micro_results["supcon"] = (supcon_weight * supcon_raw_loss).detach()
                micro_results["supcon_raw"] = supcon_raw_loss.detach()
            if entropy_weight > 0:
                micro_results["entropy"] = (-entropy_weight * entropy_raw_loss).detach()
                micro_results["entropy_raw"] = entropy_raw_loss.detach()

            if collect_artifacts:
                micro_results.update(
                    self._build_logging_payload(
                        pred_3d=pred_3d,
                        y_test=y_test,
                        true=true,
                        pre_decoder_repr_test=pre_decoder_repr_test if capture_pre_decoder_repr else None,
                        dataset_results_to_log=dataset_results_to_log,
                        dataset_results_start_index=dataset_results_start_index,
                    )
                )

        return micro_results

    def run_batch(self, dataloader_iterator):
        """Run one logical batch from streamed micro-batches and update once."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        results = {"ce": 0.0, "accuracy": 0.0}
        prior_time = 0.0

        for idx in range(self.accumulation_steps):
            with Timer() as prior_timer:
                try:
                    micro_batch = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.train_dataloader)
                    micro_batch = next(dataloader_iterator)
            prior_time += prior_timer.elapsed
            micro_batch = self._prepare_padded_batch(micro_batch)
            if micro_batch[0].shape[0] != self.config.micro_batch_size:
                raise RuntimeError(
                    f"DataLoader returned {micro_batch[0].shape[0]} datasets; "
                    f"expected micro_batch_size={self.config.micro_batch_size}"
                )
            try:
                micro_results = self.run_micro_batch(
                    micro_batch,
                    idx,
                    self.accumulation_steps,
                    collect_artifacts=False,
                    do_backward=True,
                    sync_gradients=idx == self.accumulation_steps - 1,
                    accumulation_divisor=self.accumulation_steps,
                )
                for k, v in micro_results.items():
                    if k in {
                        "confusion_matrix_sample_y_true",
                        "confusion_matrix_sample_y_pred",
                        "confusion_images",
                        "umap_images",
                    }:
                        continue
                    if k not in results:
                        results[k] = 0.0
                    results[k] += v
            except torch.cuda.OutOfMemoryError as exc:
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{self.accumulation_steps} "
                    f"at step {self.curr_step}; aborting logical step."
                )
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("A logical batch failed due to CUDA OOM") from exc

        results = self._finalize_metric_results(results, self.accumulation_steps)
        results["grad_norm"] = self._apply_optimizer_updates()
        return results, prior_time, dataloader_iterator

    def run_validation_logging_batch(self, dataloader_iterator):
        """Evaluate one logical validation batch from streamed micro-batches."""

        was_training = self.model.training
        self.model.eval()

        results = {"val_ce": 0.0, "val_accuracy": 0.0}
        prior_time = 0.0
        payload_state = {
            "confusion_y_true_payload": [],
            "confusion_y_pred_payload": [],
            "confusion_plot_payload": {},
            "umap_payload": {},
            "dataset_results_next_index": 0,
            "dataset_results_remaining": 4 if (self.master_process and self.wandb_run is not None) else 0,
        }

        with torch.no_grad():
            for idx in range(self.accumulation_steps):
                with Timer() as prior_timer:
                    try:
                        micro_batch = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(self.val_dataloader)
                        micro_batch = next(dataloader_iterator)
                prior_time += prior_timer.elapsed
                micro_batch = self._prepare_padded_batch(micro_batch)
                micro_results = self.run_micro_batch(
                    micro_batch,
                    idx,
                    self.accumulation_steps,
                    collect_artifacts=True,
                    dataset_results_to_log=payload_state["dataset_results_remaining"],
                    dataset_results_start_index=payload_state["dataset_results_next_index"],
                    do_backward=False,
                )

                if "ce" in micro_results:
                    results["val_ce"] += micro_results["ce"]
                if "accuracy" in micro_results:
                    results["val_accuracy"] += micro_results["accuracy"]

                self._merge_micro_results(results={}, micro_results=micro_results, payload_state=payload_state)

        val_artifacts = self._finalize_confusion_payload(payload_state)
        for key, value in list(val_artifacts.items()):
            val_artifacts[f"val_{key}"] = value
            del val_artifacts[key]

        results.update(val_artifacts)
        results = self._finalize_metric_results(results, self.accumulation_steps)
        if was_training:
            self.model.train()

        return results, prior_time, dataloader_iterator


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
