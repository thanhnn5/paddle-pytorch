"""Learning-rate schedule(s) used by distillation training."""

import math


def warmup_cosine_lr(step: int, total_steps: int, warmup_steps: int,
                     peak_lr: float, min_lr: float = 1e-6) -> float:
    """Linear warmup -> cosine decay to min_lr.

    Inline implementation rather than torch.optim.lr_scheduler because the
    schedule length is determined by min(epochs * steps_per_epoch, max_steps)
    which is awkward to feed into the stateful scheduler API.
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
