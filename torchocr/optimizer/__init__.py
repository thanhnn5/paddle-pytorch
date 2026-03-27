import copy

import torch

__all__ = ['build_optimizer']

# Keys consumed here; must be removed before passing to torch.optim constructor.
_GROUP_KEYS = ('backbone_lr_mult', 'neck_lr_mult', 'head_lr_mult', 'freeze_backbone')


def _build_param_groups(model, base_lr, backbone_lr_mult, neck_lr_mult, head_lr_mult):
    """
    Build PyTorch parameter groups for differential learning rates.

    Each multiplier scales relative to base_lr.  Submodules that don't exist on
    the model are silently skipped (e.g. a model with no neck).

    Returns a list of dicts suitable for torch.optim constructors.
    """
    groups = []
    assigned_ids = set()

    def _add(submodule_name, lr_mult):
        submodule = getattr(model, submodule_name, None)
        if submodule is None:
            return
        params = [p for p in submodule.parameters() if p.requires_grad and id(p) not in assigned_ids]
        if params:
            for p in params:
                assigned_ids.add(id(p))
            groups.append({'params': params, 'lr': base_lr * lr_mult})

    _add('backbone', backbone_lr_mult)
    _add('neck',     neck_lr_mult)
    _add('head',     head_lr_mult)

    # Catch any remaining trainable params (e.g. transform, or future submodules)
    remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned_ids]
    if remaining:
        groups.append({'params': remaining, 'lr': base_lr})

    return groups


def build_optimizer(optim_config, lr_scheduler_config, epochs, step_each_epoch, model):
    from . import lr as lr_module

    config = copy.deepcopy(optim_config)
    optim_name = config.pop('name')

    # --- backbone freeze (optional) ---
    freeze_backbone = config.pop('freeze_backbone', False)
    if freeze_backbone:
        backbone = getattr(model, 'backbone', None)
        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False

    # --- per-group LR (optional) ---
    backbone_lr_mult = config.pop('backbone_lr_mult', None)
    neck_lr_mult     = config.pop('neck_lr_mult',     None)
    head_lr_mult     = config.pop('head_lr_mult',     None)

    use_param_groups = any(v is not None for v in (backbone_lr_mult, neck_lr_mult, head_lr_mult))

    if use_param_groups:
        base_lr = config['lr']  # kept in config for the optimizer default
        params = _build_param_groups(
            model,
            base_lr=base_lr,
            backbone_lr_mult=backbone_lr_mult if backbone_lr_mult is not None else 1.0,
            neck_lr_mult=neck_lr_mult         if neck_lr_mult     is not None else 1.0,
            head_lr_mult=head_lr_mult         if head_lr_mult     is not None else 1.0,
        )
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())

    optim = getattr(torch.optim, optim_name)(params=params, **config)

    lr_config = copy.deepcopy(lr_scheduler_config)
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    scheduler = getattr(lr_module, lr_config.pop('name'))(**lr_config)(optimizer=optim)

    return optim, scheduler
