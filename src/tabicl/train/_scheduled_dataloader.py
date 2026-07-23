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
        batch_size_per_gp: int,
        schedule: dict[int, int | float],
        step_getter: Callable[[], int],
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size_per_gp <= 0:
            raise ValueError("batch_size_per_gp must be > 0")
        if batch_size % batch_size_per_gp != 0:
            raise ValueError("batch_size must be divisible by batch_size_per_gp")
        if not schedule:
            raise ValueError("schedule must not be empty")

        self.dataloader = dataloader
        self.batch_size = int(batch_size)
        self.batch_size_per_gp = int(batch_size_per_gp)
        self.groups_per_batch = self.batch_size // self.batch_size_per_gp
        self.step_getter = step_getter
        self.rng = random.Random(seed)

        self.schedule_steps = sorted(schedule.keys())
        self.schedule = {int(k): schedule[k] for k in self.schedule_steps}
        self._validate_schedule()
        self._validate_group_compatibility()

        self._base_iter = None
        self._delegate_forever = False
        # step -> group_id -> group_batch
        self._reservoir_by_step: dict[int, dict[int, tuple[torch.Tensor, ...]]] = {}
        self._group_id_counter = 0
        self._pending_groups: list[tuple[torch.Tensor, ...]] = []

    def __iter__(self):
        return self

    def __next__(self):
        current_step = int(self.step_getter())
        if current_step < 0:
            current_step = 0

        self._sync_with_step(current_step)

        if self._delegate_forever:
            return self._next_base_batch()

        if self._reservoir_size() == 0:
            raise RuntimeError("Reservoir is empty after schedule sync")

        sampled_groups: list[tuple[torch.Tensor, ...]] = []
        for _ in range(self.groups_per_batch):
            sampled_step = self._sample_step()
            sampled_groups.append(self._sample_group_from_step(sampled_step))

        return self._concat_groups(sampled_groups)

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

    def _validate_group_compatibility(self) -> None:
        for step in self.schedule_steps:
            size = self.schedule[step]
            if size == math.inf:
                continue
            if int(size) % self.batch_size_per_gp != 0:
                raise ValueError(
                    "Finite schedule sizes must be divisible by batch_size_per_gp for group-wise reservoir accounting"
                )

    def _active_target(self, step: int) -> tuple[int, int | float]:
        active_step = 0
        for s in self.schedule_steps:
            if s <= step:
                active_step = s
            else:
                break
        return active_step, self.schedule[active_step]

    def _sync_with_step(self, step: int) -> None:
        active_step, target_size = self._active_target(step)

        if target_size == math.inf:
            self._delegate_forever = True
            self._reservoir_by_step.clear()
            self._pending_groups.clear()
            return

        self._delegate_forever = False
        missing = int(target_size) - self._reservoir_size()
        if missing > 0:
            self._grow_reservoir(missing, reservoir_step=active_step)

    def _reservoir_size(self) -> int:
        return sum(len(step_groups) * self.batch_size_per_gp for step_groups in self._reservoir_by_step.values())

    @staticmethod
    def _group_bucket_key(group: tuple[torch.Tensor, ...]) -> tuple[int, int]:
        seq_lens = group[3]
        train_sizes = group[4]
        seq_len = int(seq_lens[0].item())
        train_size = int(train_sizes[0].item())
        if not torch.all(seq_lens == seq_len):
            raise ValueError("All datasets in a group must share seq_len")
        if not torch.all(train_sizes == train_size):
            raise ValueError("All datasets in a group must share train_size")
        return seq_len, train_size

    def _add_group(self, reservoir_step: int, group: tuple[torch.Tensor, ...]) -> None:
        # Validate homogeneity of the group metadata before adding.
        self._group_bucket_key(group)
        if reservoir_step not in self._reservoir_by_step:
            self._reservoir_by_step[reservoir_step] = {}
        self._reservoir_by_step[reservoir_step][self._group_id_counter] = group
        self._group_id_counter += 1

    def _sample_step(self) -> int:
        available_steps = [step for step, groups in self._reservoir_by_step.items() if groups]
        if not available_steps:
            raise RuntimeError("No available steps in finite reservoir")
        return self.rng.choice(available_steps)

    def _sample_group_from_step(self, step: int) -> tuple[torch.Tensor, ...]:
        groups = self._reservoir_by_step.get(step, {})
        if not groups:
            raise RuntimeError(f"Step {step} has no groups in finite reservoir")
        group_id = self.rng.choice(list(groups.keys()))
        return groups[group_id]

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

    def _grow_reservoir(self, num_items: int, reservoir_step: int) -> None:
        if num_items % self.batch_size_per_gp != 0:
            raise ValueError("Requested reservoir expansion size must be divisible by batch_size_per_gp")

        needed_groups = num_items // self.batch_size_per_gp

        while needed_groups > 0:
            while self._pending_groups and needed_groups > 0:
                self._add_group(reservoir_step=reservoir_step, group=self._pending_groups.pop())
                needed_groups -= 1
            if needed_groups == 0:
                break

            batch = self._next_base_batch()
            batch_groups = self._split_batch_into_groups(batch, self.batch_size_per_gp)
            if not batch_groups:
                continue

            take = min(needed_groups, len(batch_groups))
            for group in batch_groups[:take]:
                self._add_group(reservoir_step=reservoir_step, group=group)
            needed_groups -= take
            if take < len(batch_groups):
                self._pending_groups.extend(batch_groups[take:])

    @staticmethod
    def _split_batch_into_groups(batch, group_size: int) -> list[tuple[torch.Tensor, ...]]:
        fields = list(batch)
        prepared_fields: list[torch.Tensor] = []
        for tensor in fields:
            if isinstance(tensor, list):
                prepared_fields.append(tensor)
            else:
                prepared_fields.append(tensor.to_padded_tensor(padding=0.0) if tensor.is_nested else tensor)

        batch_size = int(prepared_fields[0].shape[0])
        if batch_size % group_size != 0:
            raise ValueError("Base dataloader batch size must be divisible by batch_size_per_gp")

        groups: list[tuple[torch.Tensor, ...]] = []
        for start in range(0, batch_size, group_size):
            end = start + group_size
            groups.append(
                tuple(field[start:end] if isinstance(field, list) else field[start:end] for field in prepared_fields)
            )
        return groups

    @staticmethod
    def _concat_groups(groups: list[tuple[torch.Tensor, ...]]):
        if not groups:
            raise RuntimeError("Cannot concatenate an empty group list")

        max_seq = max(group[0].shape[1] for group in groups)
        max_feat = max(group[0].shape[2] for group in groups)

        x_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        d_parts: list[torch.Tensor] = []
        seq_parts: list[torch.Tensor] = []
        train_parts: list[torch.Tensor] = []

        graph_parts = []
        graph_mode = len(groups[0]) == 6
        for group in groups:
            if graph_mode:
                x, y, d, seq_len, train_size, graphs = group
                graph_parts.extend(graphs)
            else:
                x, y, d, seq_len, train_size = group

            if x.shape[1] < max_seq:
                x = torch.nn.functional.pad(x, (0, 0, 0, max_seq - x.shape[1]))
                y = torch.nn.functional.pad(y, (0, max_seq - y.shape[1]))
            if x.shape[2] < max_feat:
                x = torch.nn.functional.pad(x, (0, max_feat - x.shape[2], 0, 0))

            x_parts.append(x)
            y_parts.append(y)
            d_parts.append(d)
            seq_parts.append(seq_len)
            train_parts.append(train_size)

        result = (
            torch.cat(x_parts, dim=0),
            torch.cat(y_parts, dim=0),
            torch.cat(d_parts, dim=0),
            torch.cat(seq_parts, dim=0),
            torch.cat(train_parts, dim=0),
        )
        return result + (graph_parts,) if graph_mode else result
