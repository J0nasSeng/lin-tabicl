import torch
import pytest

from src.tabicl.train._scheduled_dataloader import ScheduledDataLoader


class _BatchStream:
    def __init__(self, batches):
        self.batches = batches
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.idx]
        self.idx += 1
        return batch


class _StepState:
    def __init__(self, value: int = 0):
        self.value = value

    def __call__(self) -> int:
        return self.value


def _make_batch(seq_lens, train_sizes, feat_dim=3, max_seq=6):
    batch_size = len(seq_lens)
    x = torch.randn(batch_size, max_seq, feat_dim)
    y = torch.randint(0, 3, (batch_size, max_seq))
    d = torch.full((batch_size,), feat_dim, dtype=torch.long)
    seq = torch.tensor(seq_lens, dtype=torch.long)
    train = torch.tensor(train_sizes, dtype=torch.long)
    return x, y, d, seq, train


def test_scheduled_loader_requires_group_divisibility():
    stream = _BatchStream([_make_batch([5, 5], [2, 2])])

    with pytest.raises(ValueError, match="divisible"):
        ScheduledDataLoader(
            dataloader=stream,
            batch_size=3,
            batch_size_per_gp=2,
            schedule={0: 4, 10: float("inf")},
            step_getter=lambda: 0,
            seed=0,
        )


def test_scheduled_loader_requires_schedule_group_compatibility():
    stream = _BatchStream([_make_batch([5, 5], [2, 2])])

    with pytest.raises(ValueError, match="Finite schedule sizes must be divisible"):
        ScheduledDataLoader(
            dataloader=stream,
            batch_size=4,
            batch_size_per_gp=2,
            schedule={0: 3, 10: float("inf")},
            step_getter=lambda: 0,
            seed=0,
        )


def test_step_zero_initializes_step_indexed_reservoir():
    stream = _BatchStream([_make_batch([5, 5, 6, 6], [2, 2, 3, 3])])
    step_state = _StepState(0)
    loader = ScheduledDataLoader(
        dataloader=stream,
        batch_size=4,
        batch_size_per_gp=2,
        schedule={0: 4, 10: float("inf")},
        step_getter=step_state,
        seed=1,
    )

    _ = next(loader)

    assert loader._reservoir_size() == 4
    assert 0 in loader._reservoir_by_step
    assert len(loader._reservoir_by_step[0]) == 2


def test_milestone_expansion_adds_groups_under_new_step():
    stream = _BatchStream([_make_batch([5, 5, 6, 6], [2, 2, 3, 3])])
    step_state = _StepState(0)
    loader = ScheduledDataLoader(
        dataloader=stream,
        batch_size=4,
        batch_size_per_gp=2,
        schedule={0: 2, 5: 4, 10: float("inf")},
        step_getter=step_state,
        seed=2,
    )

    _ = next(loader)
    assert set(loader._reservoir_by_step.keys()) == {0}
    assert loader._reservoir_size() == 2

    step_state.value = 5
    _ = next(loader)
    assert set(loader._reservoir_by_step.keys()) == {0, 5}
    assert len(loader._reservoir_by_step[0]) == 1
    assert len(loader._reservoir_by_step[5]) == 1
    assert loader._reservoir_size() == 4


def test_step_then_group_sampling_keeps_groupwise_blocks_and_uses_multiple_steps():
    # Step 0 stores one group with seq=5, step 5 stores one group with seq=6.
    stream = _BatchStream([_make_batch([5, 5, 6, 6], [2, 2, 3, 3])])
    step_state = _StepState(0)
    loader = ScheduledDataLoader(
        dataloader=stream,
        batch_size=2,
        batch_size_per_gp=2,
        schedule={0: 2, 5: 4, 10: float("inf")},
        step_getter=step_state,
        seed=3,
    )

    _ = next(loader)
    step_state.value = 5
    _ = next(loader)

    counts = {5: 0, 6: 0}
    for _ in range(200):
        batch = next(loader)
        _, _, _, seq_lens, train_sizes = batch

        # Group-wise sampling means both elements in this batch come from one group.
        assert int(seq_lens[0].item()) == int(seq_lens[1].item())
        assert int(train_sizes[0].item()) == int(train_sizes[1].item())

        sampled_seq = int(seq_lens[0].item())
        counts[sampled_seq] += 1

    assert counts[5] > 0
    assert counts[6] > 0

    frac = counts[5] / (counts[5] + counts[6])
    assert 0.30 <= frac <= 0.70


def test_scheduled_loader_inf_delegates_directly():
    base_batch = _make_batch([5, 5, 5, 5], [2, 2, 2, 2])
    stream = _BatchStream([base_batch])
    loader = ScheduledDataLoader(
        dataloader=stream,
        batch_size=4,
        batch_size_per_gp=2,
        schedule={0: float("inf")},
        step_getter=_StepState(0),
        seed=0,
    )

    delegated = next(loader)
    for got, exp in zip(delegated, base_batch):
        assert torch.equal(got, exp)
