# -*- coding: utf-8 -*-
# @Time    : 2023/8/27 0:08
# @Author  : zhoujun
import math
from functools import partial
from torch.optim import lr_scheduler


class StepLR(object):
    def __init__(self,
                 step_each_epoch,
                 step_size,
                 warmup_epoch=0,
                 gamma=0.1,
                 last_epoch=-1,
                 **kwargs):
        super(StepLR, self).__init__()
        self.step_size = step_each_epoch * step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return self.gamma ** (current_step // self.step_size)


class MultiStepLR(object):
    def __init__(self,
                 step_each_epoch,
                 milestones,
                 warmup_epoch=0,
                 gamma=0.1,
                 last_epoch=-1,
                 **kwargs):
        super(MultiStepLR, self).__init__()
        self.milestones = [step_each_epoch * e for e in milestones]
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return self.gamma ** len([m for m in self.milestones if m <= current_step])

class ConstLR(object):
    def __init__(self,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(ConstLR, self).__init__()
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1.0, self.warmup_epoch))
        return 1.0


class LinearLR(object):
    def __init__(self,
                 epochs,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(LinearLR, self).__init__()
        self.epochs = epochs * step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return max(0.0, float(self.epochs - current_step) / float(max(1, self.epochs - self.warmup_epoch)))


class CosineAnnealingLR(object):
    def __init__(self,
                 epochs,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(CosineAnnealingLR, self).__init__()
        self.epochs = epochs * step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step, num_cycles=0.5):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        progress = float(current_step - self.warmup_epoch) / float(max(1, self.epochs - self.warmup_epoch))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


class FlatCosineLR(object):
    """Warmup -> flat hold at peak LR -> cosine decay -> optional min-flat tail.

    Mirrors the FlatCosine schedule used by DEIM / RT-DETRv4 (see
    engine/optim/lr_scheduler.py in arxiv 2510.25257 reference repo). Four
    phases stitched together by `lambda_func`:

      1. Warmup        (steps 0 .. warmup_iter):
                       quadratic ramp from 0 to peak_lr.
                       multiplier(step) = (step / warmup_iter) ** 2
      2. Flat hold     (warmup_iter .. flat_iter):
                       constant at peak_lr.
                       multiplier(step) = 1.0
      3. Cosine decay  (flat_iter .. total_iter - cooldown_iter):
                       half-cosine from peak_lr to peak_lr * lr_gamma.
                       multiplier(step) = lr_gamma + (1 - lr_gamma) * cos_decay
      4. Min-flat tail (last cooldown_iter steps):
                       constant at peak_lr * lr_gamma.
                       multiplier(step) = lr_gamma

    Rationale per the DEIM paper: DETR-style detectors converge faster when
    given a long flat plateau at peak LR before cosine decay starts (most
    optimization happens during this phase). The cooldown tail at min LR is
    optional — useful for end-of-training stability (final epochs converge at
    a steady low LR) or to align with an augmentation-off finetune phase if
    you have one. Default `cooldown_epoch=0` means the schedule reduces to
    `warmup -> flat -> cosine-to-min` with the last step landing at `lr_gamma`.

    Args:
        epochs: total number of training epochs.
        step_each_epoch: number of iterations per epoch.
        warmup_epoch: epochs of quadratic warmup (default 2).
        flat_epoch: epoch index at which cosine decay begins (default = half
            of `epochs`). Must satisfy `warmup_epoch <= flat_epoch <= epochs`.
        cooldown_epoch: final epochs to hold at min LR (default 0). Setting
            > 0 reserves the last N epochs as a steady-min-LR cooldown
            instead of letting cosine decay run all the way to the end.
        lr_gamma: minimum LR as a fraction of peak (default 0.5 = decay to
            50% of peak, matching RT-DETRv4). Use 0.05-0.1 for a deeper
            decay tail.
        last_epoch: passed to torch's LambdaLR for resume support.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 warmup_epoch=2,
                 flat_epoch=None,
                 cooldown_epoch=0,
                 lr_gamma=0.5,
                 last_epoch=-1,
                 **kwargs):
        super(FlatCosineLR, self).__init__()
        if flat_epoch is None:
            flat_epoch = epochs // 2

        if not (0 <= warmup_epoch <= flat_epoch <= epochs):
            raise ValueError(
                f"FlatCosineLR requires 0 <= warmup_epoch ({warmup_epoch}) "
                f"<= flat_epoch ({flat_epoch}) <= epochs ({epochs})"
            )
        if cooldown_epoch < 0 or cooldown_epoch >= epochs - flat_epoch:
            raise ValueError(
                f"cooldown_epoch ({cooldown_epoch}) must be in [0, epochs-flat_epoch={epochs-flat_epoch})"
            )
        if not (0.0 <= lr_gamma < 1.0):
            raise ValueError(f"lr_gamma must be in [0, 1), got {lr_gamma}")

        self.total_iter = epochs * step_each_epoch
        self.warmup_iter = warmup_epoch * step_each_epoch
        self.flat_iter = flat_epoch * step_each_epoch
        self.cooldown_iter = cooldown_epoch * step_each_epoch
        self.lr_gamma = float(lr_gamma)
        self.last_epoch = last_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        # Phase 1: quadratic warmup
        if current_step < self.warmup_iter:
            return (current_step / max(1, self.warmup_iter)) ** 2

        # Phase 2: flat hold at peak
        if current_step <= self.flat_iter:
            return 1.0

        # Phase 4: min-flat tail
        if self.cooldown_iter > 0 and current_step >= self.total_iter - self.cooldown_iter:
            return self.lr_gamma

        # Phase 3: half-cosine decay from peak -> peak * lr_gamma
        decay_span = max(1, self.total_iter - self.flat_iter - self.cooldown_iter)
        progress = (current_step - self.flat_iter) / decay_span
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
        return self.lr_gamma + (1.0 - self.lr_gamma) * cos_factor


class PolynomialLR(object):
    def __init__(self,
                 step_each_epoch,
                 epochs,
                 lr_end=1e-7,
                 power=1.0,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(PolynomialLR, self).__init__()
        self.lr_end = lr_end
        self.power = power
        self.epochs = epochs * step_each_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch
        self.last_epoch = last_epoch

    def __call__(self, optimizer):
        lr_lambda = partial(
            self.lambda_func,
            lr_init=optimizer.defaults["lr"],
        )
        return lr_scheduler.LambdaLR(optimizer, lr_lambda, self.last_epoch)

    def lambda_func(self, current_step, lr_init):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        elif current_step > self.epochs:
            return self.lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - self.lr_end
            decay_steps = self.epochs - self.warmup_epoch
            pct_remaining = 1 - (current_step - self.warmup_epoch) / decay_steps
            decay = lr_range * pct_remaining ** self.power + self.lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init