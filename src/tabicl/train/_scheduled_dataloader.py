from __future__ import annotations

from typing import Callable
import math
import random

import torch


def parse_step_size_schedule(steps_csv: str, sizes_csv: str) -> dict[int, int | float]:
    """Parse and validate a step->reservoir-size schedule from CLI CSV args.

    Parameters
    ----------
    steps_csv : str
        Comma-separated integer steps. Must start with 0 and be strictly increasing.

    sizes_csv : str
        Comma-separated sizes where each item is a positive integer or "inf".
        At least one "inf" entry is required.

    Returns
    -------
    dict[int, int | float]
        Mapping from step to reservoir size where "inf" is represented as ``math.inf``.
    """

    step_tokens = [tok.strip() for tok in steps_csv.split(",") if tok.strip()]
    size_tokens = [tok.strip().lower() for tok in sizes_csv.split(",") if tok.strip()]

    if not step_tokens:
        raise ValueError("Schedule steps cannot be empty")
    if not size_tokens:
        raise ValueError("Schedule sizes cannot be empty")
    if len(step_tokens) != len(size_tokens):
        raise ValueError("Schedule steps and sizes must have the same length")

    steps: list[int] = []
    for token in step_tokens:
        try:
            step = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid schedule step: {token}") from exc
        if step < 0:
            raise ValueError("Schedule steps must be >= 0")
        steps.append(step)

    if steps[0] != 0:
        raise ValueError("Schedule must start at step 0")
    if any(curr <= prev for prev, curr in zip(steps, steps[1:])):
        raise ValueError("Schedule steps must be strictly increasing")

    sizes: list[int | float] = []
    for token in size_tokens:
        if token == "inf":
            sizes.append(math.inf)
            continue
        try:
            size = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid schedule size: {token}") from exc
        if size <= 0:
            raise ValueError("Finite schedule sizes must be positive")
        sizes.append(size)

    if not any(size == math.inf for size in sizes):
        raise ValueError('Schedule must include at least one "inf" size')

    for prev_size, next_size in zip(sizes, sizes[1:]):
        if prev_size == math.inf and next_size != math.inf:
            raise ValueError('Schedule cannot switch from "inf" back to a finite size')
        if prev_size != math.inf and next_size != math.inf and next_size < prev_size:
            raise ValueError("Finite schedule sizes must be non-decreasing")

    return {step: size for step, size in zip(steps, sizes)}


class ScheduledDataLoader:
    """Wrap a dataloader with a scheduled reservoir of datasets.

    The schedule is a dictionary mapping training step to reservoir size.
    At each call, returns ``batch_size`` datasets randomly sampled from the
    reservoir. When the active size is ``inf``, this loader delegates directly
    to the wrapped dataloader and no longer uses reservoir sampling.
    """

    def __init__(
        self,
        dataloader,
        batch_size: int,
        schedule: dict[int, int | float],
        step_getter: Callable[[], int],
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not schedule:
            raise ValueError("schedule must not be empty")

        self.dataloader = dataloader
        self.batch_size = int(batch_size)
        self.step_getter = step_getter
        self.rng = random.Random(seed)

        self.schedule_steps = sorted(schedule.keys())
        self.schedule = {int(k): schedule[k] for k in self.schedule_steps}
        self._validate_schedule()

        self._base_iter = None
        self._initialized = False
        self._delegate_forever = False
        self._reservoir_by_key: dict[tuple[int, int], list[tuple[torch.Tensor, ...]]] = {}
        self._pending_items: list[tuple[torch.Tensor, ...]] = []

    def __iter__(self):
        return self

    def __next__(self):
        current_step = int(self.step_getter())
        if current_step < 0:
            current_step = 0

        self._sync_with_step(current_step)

        if self._delegate_forever:
            return self._next_base_batch()

        if not self._reservoir_by_key:
            raise RuntimeError("Reservoir is empty after schedule sync")

        keys = list(self._reservoir_by_key.keys())
        key_weights = [len(self._reservoir_by_key[k]) for k in keys]
        sampled_key = self.rng.choices(keys, weights=key_weights, k=1)[0]
        bucket = self._reservoir_by_key[sampled_key]
        sampled_items = [bucket[self.rng.randrange(len(bucket))] for _ in range(self.batch_size)]
        return self._stack_items(sampled_items)

    def _validate_schedule(self) -> None:
        if self.schedule_steps[0] != 0:
            raise ValueError("schedule must start at step 0")

        if any(curr <= prev for prev, curr in zip(self.schedule_steps, self.schedule_steps[1:])):
            raise ValueError("schedule steps must be strictly increasing")

        sizes = [self.schedule[s] for s in self.schedule_steps]
        if not any(size == math.inf for size in sizes):
            raise ValueError('schedule must include at least one "inf" size')

        for prev_size, next_size in zip(sizes, sizes[1:]):
            if prev_size == math.inf and next_size != math.inf:
                raise ValueError('schedule cannot switch from "inf" back to finite')
            if prev_size != math.inf and next_size != math.inf and next_size < prev_size:
                raise ValueError("finite schedule sizes must be non-decreasing")

    def _active_size(self, step: int) -> int | float:
        active_step = 0
        for s in self.schedule_steps:
            if s <= step:
                active_step = s
            else:
                break
        return self.schedule[active_step]

    def _sync_with_step(self, step: int) -> None:
        target_size = self._active_size(step)

        if target_size == math.inf:
            self._delegate_forever = True
            self._initialized = True
            self._reservoir_by_key.clear()
            self._pending_items.clear()
            return

        self._delegate_forever = False
        self._initialized = True
        missing = int(target_size) - self._reservoir_size()
        if missing > 0:
            self._grow_reservoir(missing)

    def _reservoir_size(self) -> int:
        return sum(len(bucket) for bucket in self._reservoir_by_key.values())

    @staticmethod
    def _item_bucket_key(item: tuple[torch.Tensor, ...]) -> tuple[int, int]:
        seq_len = int(item[3].item())
        train_size = int(item[4].item())
        return seq_len, train_size

    def _add_to_reservoir(self, item: tuple[torch.Tensor, ...]) -> None:
        key = self._item_bucket_key(item)
        if key not in self._reservoir_by_key:
            self._reservoir_by_key[key] = []
        self._reservoir_by_key[key].append(item)

    def _ensure_base_iter(self) -> None:
        if self._base_iter is None:
            self._base_iter = iter(self.dataloader)

    def _next_base_batch(self):
        self._ensure_base_iter()
        try:
            return next(self._base_iter)
        except StopIteration:
            self._base_iter = iter(self.dataloader)
            return next(self._base_iter)

    def _grow_reservoir(self, num_items: int) -> None:
        needed = num_items

        while needed > 0:
            while self._pending_items and needed > 0:
                self._add_to_reservoir(self._pending_items.pop())
                needed -= 1
            if needed == 0:
                break

            batch = self._next_base_batch()
            batch_items = self._split_batch_into_items(batch)
            if not batch_items:
                continue

            take = min(needed, len(batch_items))
            for item in batch_items[:take]:
                self._add_to_reservoir(item)
            needed -= take
            if take < len(batch_items):
                self._pending_items.extend(batch_items[take:])

    @staticmethod
    def _split_batch_into_items(batch) -> list[tuple[torch.Tensor, ...]]:
        fields = list(batch)
        prepared_fields: list[torch.Tensor] = []
        for tensor in fields:
            prepared_fields.append(tensor.to_padded_tensor(padding=0.0) if tensor.is_nested else tensor)

        batch_size = int(prepared_fields[0].shape[0])
        return [tuple(field[idx] for field in prepared_fields) for idx in range(batch_size)]

    @staticmethod
    def _stack_items(items: list[tuple[torch.Tensor, ...]]):
        transposed = list(zip(*items))
        return tuple(torch.stack(list(collected), dim=0) for collected in transposed)
