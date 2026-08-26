from types import SimpleNamespace

from src.tabicl.prior._dataset import PriorDataset
from src.tabicl.train._run import Trainer


def _trainer_for_loader_tests():
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        prior_device="cpu",
        device="cpu",
    )
    return trainer


def test_train_and_validation_loader_resource_policies_are_separate():
    trainer = _trainer_for_loader_tests()
    train_dataset = iter(())
    val_dataset = iter(())

    train_loader = trainer._build_train_prior_dataloader(train_dataset)
    val_loader = trainer._build_validation_prior_dataloader(val_dataset)

    assert train_loader.num_workers == 4
    assert train_loader.prefetch_factor == 64
    assert train_loader.pin_memory
    assert train_loader.persistent_workers

    assert val_loader.num_workers == 0
    assert val_loader.prefetch_factor is None
    assert not val_loader.pin_memory
    assert not val_loader.persistent_workers


def test_prior_dataset_stream_item_is_one_homogeneous_group():
    dataset = PriorDataset(
        batch_size=4,
        batch_size_per_gp=4,
        min_features=2,
        max_features=4,
        max_classes=3,
        min_seq_len=8,
        max_seq_len=9,
        prior_type="dummy",
        graph_backend=False,
        device="cpu",
    )

    batch = next(iter(dataset))
    x, y, d, seq_lens, train_sizes = batch

    assert x.shape[0] == 4
    assert y.shape[0] == 4
    assert d.shape == (4,)
    assert seq_lens.shape == (4,)
    assert train_sizes.shape == (4,)
    assert d.unique().numel() == 1
    assert seq_lens.unique().numel() == 1
    assert train_sizes.unique().numel() == 1
