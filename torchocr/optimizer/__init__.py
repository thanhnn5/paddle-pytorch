import copy
import warnings

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

    # --- Extract group keys (Issue 1: use loop instead of manual pops) ---
    group_vals = {}
    for key in _GROUP_KEYS:
        group_vals[key] = config.pop(key, None)

    freeze_backbone = group_vals['freeze_backbone'] if group_vals['freeze_backbone'] is not None else False
    backbone_lr_mult = group_vals['backbone_lr_mult']
    neck_lr_mult     = group_vals['neck_lr_mult']
    head_lr_mult     = group_vals['head_lr_mult']

    # --- backbone freeze (optional) ---
    if freeze_backbone:
        backbone = getattr(model, 'backbone', None)
        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False

    use_param_groups = any(v is not None for v in (backbone_lr_mult, neck_lr_mult, head_lr_mult))

    # --- Issue 2: PolynomialLR incompatibility guard ---
    if use_param_groups:
        scheduler_name = lr_scheduler_config.get('name')
        if scheduler_name == 'PolynomialLR':
            raise ValueError(
                "PolynomialLR is incompatible with per-group learning rates (backbone_lr_mult / "
                "neck_lr_mult / head_lr_mult) because it anchors decay on optimizer.defaults['lr'] "
                "rather than each group's individual lr. Use a different scheduler or remove the "
                "*_lr_mult keys."
            )

    # --- Issue 3: freeze_backbone + backbone_lr_mult warning ---
    if freeze_backbone and backbone_lr_mult is not None:
        warnings.warn(
            "freeze_backbone=True and backbone_lr_mult is set: backbone params are frozen so backbone_lr_mult has no effect.",
            UserWarning,
            stacklevel=2,
        )

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
