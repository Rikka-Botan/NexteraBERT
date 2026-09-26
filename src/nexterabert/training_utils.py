"""Distributed-training helpers shared by the pretraining / fine-tuning scripts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistEnv:
    """Initialise ``torch.distributed`` from the env vars ``torchrun`` exports."""
    if int(os.environ.get("RANK", -1)) != -1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            # Bind the process group to this rank's device. Without device_id NCCL
            # infers the device for barrier()/collectives from whatever context is
            # current — which warns, and picks wrong if the current device ever
            # differs from local_rank. CPU/gloo takes no device_id.
            kwargs = {"device_id": device} if backend == "nccl" else {}
            dist.init_process_group(backend=backend, **kwargs)
        return DistEnv(True, rank, local_rank, world_size, device)

    # single process
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return DistEnv(False, 0, 0, 1, device)


def cleanup_distributed(env: DistEnv):
    if env.is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def all_reduce_mean(value: torch.Tensor, env: DistEnv) -> torch.Tensor:
    if env.is_distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / env.world_size
    return value


class _WarmupSchedule:
    """Shared plumbing for the pretraining LR schedules below.

    Subclasses implement ``_factor(step) -> multiplier on the peak LR``.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0
        # Start the groups at the step-0 factor (0 with warmup) instead of the
        # peak LR — otherwise the very first optimizer step, taken before the
        # first .step() call, would bypass the warmup entirely.
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * self._factor(0)

    def _factor(self, step):
        raise NotImplementedError

    def step(self):
        self._step += 1
        factor = self._factor(self._step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class WarmupStableDecay(_WarmupSchedule):
    """ModernBERT pretraining LR schedule: trapezoidal (WSD) with 1-sqrt decay.

    Linear warmup, then the peak LR held CONSTANT for the bulk of the run,
    then a ``1 - sqrt`` decay over the final ``decay_steps``.  ModernBERT
    (Warner et al., 2024, "Learning Rate Schedule") uses this Warmup-Stable-
    Decay shape over cosine because it lets any checkpoint be continued
    without a cold restart, and picks 1-sqrt decay (Hagele et al., 2024)
    having "found it to outperform linear and cosine decay".

    ``decay_steps=0`` gives the pure warmup+constant leg, which is what
    ModernBERT-base's 1024-token phase runs — Table 3 lists no decay tokens
    for it, the whole LR decay being spent at the very end of the 8192-token
    phase instead.  Because the LR never anneals, a phase-1 checkpoint is a
    hand-off to phase 2, not a finished model.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, decay_steps=0,
                 min_lr_ratio=0.0):
        self.decay_steps = max(0, min(decay_steps, total_steps - warmup_steps))
        super().__init__(optimizer, warmup_steps, total_steps, min_lr_ratio)

    def _factor(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        decay_start = self.total_steps - self.decay_steps
        if self.decay_steps <= 0 or step < decay_start:
            return 1.0
        progress = min((step - decay_start) / max(1, self.decay_steps), 1.0)
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * (1.0 - math.sqrt(progress))


class WarmupCosineDecay(_WarmupSchedule):
    """OptiBERT / NeoBERT pretraining LR schedule.

    Linear warmup, then cosine decay from the peak LR down to
    ``min_lr_ratio`` (10%) of it over the remaining steps ("decayed to 10%
    of its peak value following a cosine schedule", OptiBERT Appendix A).
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        super().__init__(optimizer, warmup_steps, total_steps, min_lr_ratio)

    def _factor(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine


def save_checkpoint(path, model, optimizer, scheduler, step, config_dict, extra=None):
    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)
    payload = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_step": getattr(scheduler, "_step", 0),
        "step": step,
        "config": config_dict,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_all_but(model, patterns) -> dict:
    """Keep only the parameters whose name contains one of ``patterns`` trainable.

    Every other parameter gets ``requires_grad=False``. Parameters the model had
    already frozen (see ``freeze_pretraining_unused``) stay frozen even if they
    match. Must run BEFORE the model is wrapped in DDP / compiled: DDP forbids
    changing ``requires_grad`` afterwards, and ``build_optimizer`` reads the flag
    when it collects its groups. Returns a small summary for the log.
    """
    patterns = [str(p) for p in patterns if str(p)]
    kept, kept_numel, frozen = [], 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(pat in name for pat in patterns):
            kept.append(name)
            kept_numel += p.numel()
        else:
            p.requires_grad_(False)
            frozen += 1
    return {"trainable_tensors": len(kept), "trainable_params": kept_numel,
            "frozen_tensors": frozen, "names": kept}
