from argparse import Namespace

from src.tabicl.train._run import Trainer
from tabicl._model.nanotabicl import NanoTabICLv2


def _minimal_config() -> Namespace:
    return Namespace(
        # Trainer/runtime
        device="cpu",
        wandb_log=False,
        checkpoint_dir=None,
        wandb_dir=None,
        wandb_project="TabICL",
        wandb_name=None,
        wandb_id=None,
        wandb_mode="offline",
        max_steps=1,
        # model selection
        model_type="nanotabicl",
        # architecture
        max_classes=5,
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        row_rope_base=100000.0,
        icl_num_blocks=1,
        icl_nhead=2,
        icl_backend="graph",
        icl_decoder_type="mlp",
        icl_soft_kmeans_temperature=0.1,
        graph_min_train_neighbors=1,
        graph_max_train_neighbors=2,
        graph_same_label_ratio=0.9,
        graph_cross_label_ratio=0.1,
        graph_test_k_per_class=1,
        graph_seed=None,
        graph_share_across_batch=False,
        graph_share_require_identical_labels=True,
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        # freeze flags
        freeze_col=False,
        freeze_row=False,
        freeze_icl=False,
        # compile/ddp/amp
        model_compile=False,
        amp=False,
        dtype="float32",
        # optimizer/scheduler
        lr=1e-3,
        weight_decay=0.0,
        scheduler="cosine_warmup",
        warmup_proportion=0.0,
        warmup_steps=0,
        cosine_num_cycles=1,
        cosine_amplitude_decay=1.0,
        cosine_lr_end=0.0,
        poly_decay_lr_end=1e-7,
        poly_decay_power=1.0,
        gradient_clipping=0.0,
        # prior/dataloader placeholders
        prior_dir=None,
        load_prior_start=0,
        delete_after_load=False,
        batch_size=2,
        micro_batch_size=1,
        batch_size_per_gp=1,
        min_features=2,
        max_features=5,
        min_seq_len=4,
        max_seq_len=4,
        log_seq_len=False,
        seq_len_per_gp=False,
        min_train_size=0.4,
        max_train_size=0.6,
        replay_small=False,
        prior_type="nanotabicl",
        prior_device="cpu",
        # misc
        np_seed=0,
        torch_seed=0,
        log_conf_mat_every=0,
        supcon_weight=0.0,
        entropy_weight=0.0,
        # checkpoint loading
        checkpoint_path=None,
        only_load_model=False,
        save_temp_every=100,
        save_perm_every=1000,
        max_checkpoints=0,
    )


def test_trainer_builds_nanotabicl_model():
    cfg = _minimal_config()

    trainer = Trainer(cfg)

    assert trainer.model_type == "nanotabicl"
    assert isinstance(trainer.raw_model, NanoTabICLv2)
