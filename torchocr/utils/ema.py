"""Exponential Moving Average of model weights.

Port of PaddleOCR ``ppocr/utils/ema.py::ModelEMA`` (enabled by
``Global.use_ema`` in the PP-OCRv6 det configs). Same semantics:

    shadow = decay * shadow + (1 - decay) * weight     (fp32, starts at 0)
    decay  = min(decay, (1 + step) / (10 + step))      ('threshold', default)
    apply() returns shadow / (1 - last_decay ** step)  (bias-corrected)

torch-specific: non-floating-point buffers (BatchNorm ``num_batches_tracked``,
which Paddle doesn't have) are not averaged; they are copied from the live
model on every update.
"""
import math

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model, decay=0.9998, gamma=2000, ema_decay_type='threshold',
                 ema_filter_no_grad=False):
        self.decay = decay
        self.gamma = gamma
        self.ema_decay_type = ema_decay_type
        self.step = 0
        self._decay = decay

        # Frozen params (requires_grad=False, e.g. a distillation teacher) pass
        # through unchanged when ema_filter_no_grad. BN running stats are
        # buffers, not parameters, so they are never black-listed.
        self.ema_black_list = set()
        if ema_filter_no_grad:
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    self.ema_black_list.add(n)

        self.state_dict = {}
        for k, v in model.state_dict().items():
            v = v.detach()
            if k in self.ema_black_list or not v.is_floating_point():
                self.state_dict[k] = v.clone()
            else:
                self.state_dict[k] = torch.zeros_like(v, dtype=torch.float32)

    def _get_decay(self):
        if self.ema_decay_type == 'threshold':
            return min(self.decay, (1 + self.step) / (10 + self.step))
        elif self.ema_decay_type == 'exponential':
            return self.decay * (1 - math.exp(-(self.step + 1) / self.gamma))
        return self.decay  # 'normal'

    def _averaged(self, k, v):
        return k not in self.ema_black_list and v.is_floating_point()

    @torch.no_grad()
    def update(self, model):
        decay = self._get_decay()
        self._decay = decay
        model_dict = model.state_dict()
        for k, v in self.state_dict.items():
            if k not in model_dict:
                continue
            cur = model_dict[k].detach()
            if self._averaged(k, v):
                v.mul_(decay).add_(cur.float(), alpha=1 - decay)
            elif k not in self.ema_black_list:
                v.copy_(cur)
        self.step += 1

    @torch.no_grad()
    def apply(self):
        """Bias-corrected EMA state dict for eval/save. Does not modify state."""
        if self.step == 0:
            return {k: v.clone() for k, v in self.state_dict.items()}
        state = {}
        for k, v in self.state_dict.items():
            if self._averaged(k, v) and self.ema_decay_type != 'exponential':
                v = v / (1 - self._decay ** self.step)
            state[k] = v.clone()
        return state

    def state_dict_for_save(self):
        return {'ema_state': self.state_dict, 'step': self.step, 'decay': self._decay}

    def set_state_dict(self, d):
        self.state_dict = {k: v.clone() for k, v in d['ema_state'].items()}
        self.step = d['step']
        self._decay = d.get('decay', self._get_decay())

    def to(self, device):
        self.state_dict = {k: v.to(device) for k, v in self.state_dict.items()}
        return self
