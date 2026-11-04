"""Tests for the PP-OCRv6 det training pieces ported from PaddleOCR:
DiceFocalLoss / aux DBLoss, fix_nan head, ModelEMA, RandomCrop,
RandomPerspective, ColorJitter (cv2 backend), IaaAugment albumentations
backend and the epoch-aware MakeShrinkMap / MakeBorderMap.
"""
import copy
import os
import pickle
import random
import tempfile

import numpy as np
import pytest
import torch
import yaml

V6_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs', 'det', 'PP-OCRv6')
V6_SIZES = ['tiny', 'small', 'medium']


def _v6_cfg(size):
    return yaml.safe_load(open(os.path.join(V6_DIR, f'PP-OCRv6_{size}_det.yml')))


def _seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _det_sample(h=480, w=600):
    """Synthetic det sample in the format DetLabelEncode produces."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    polys = np.array([
        [[50, 60], [250, 60], [250, 110], [50, 110]],
        [[300, 200], [550, 200], [550, 260], [300, 260]],
        [[100, 350], [200, 350], [200, 380], [100, 380]],
    ], dtype=np.float32)
    return {'image': img, 'polys': polys, 'ignore_tags': [False, False, True], 'texts': ['a', 'b', '###']}


def _labels(b=2, h=64, w=64):
    shrink = (torch.rand(b, h, w) > 0.8).float()
    return [
        torch.zeros(b, 3, h, w),
        torch.rand(b, h, w),           # threshold_map
        torch.ones(b, h, w),           # threshold_mask
        shrink,                        # shrink_map
        torch.ones(b, h, w),           # shrink_mask
    ]


# --------------------------------------------------------------------- losses
def test_masked_focal_loss_matches_torchvision_sigmoid_focal_loss():
    from torchvision.ops import sigmoid_focal_loss
    from torchocr.losses.det_basic_loss import MaskedFocalLoss

    _seed()
    pred = torch.rand(2, 32, 32).clamp(0.01, 0.99)
    gt = (torch.rand(2, 32, 32) > 0.7).float()
    mask = (torch.rand(2, 32, 32) > 0.2).float()

    got = MaskedFocalLoss(alpha=0.25, gamma=2.5, eps=1e-6)(pred, gt, mask)

    logit = torch.log(pred / (1 - pred))
    ref = sigmoid_focal_loss(logit, gt, alpha=0.25, gamma=2.5, reduction='none')
    ref = (ref * mask).sum() / (mask.sum() + 1e-6)
    assert torch.allclose(got, ref, atol=1e-6)


def test_dice_focal_loss_is_weighted_sum():
    from torchocr.losses.det_basic_loss import DiceFocalLoss, DiceLoss, MaskedFocalLoss

    _seed()
    pred = torch.rand(2, 16, 16)
    gt = (torch.rand(2, 16, 16) > 0.5).float()
    mask = torch.ones(2, 16, 16)
    got = DiceFocalLoss(dice_weight=0.7, focal_weight=1.3, focal_alpha=0.25, focal_gamma=2.0)(pred, gt, mask)
    ref = 0.7 * DiceLoss()(pred, gt, mask) + 1.3 * MaskedFocalLoss(0.25, 2.0)(pred, gt, mask)
    assert torch.allclose(got, ref)


def test_dbloss_aux_terms_added_with_weights():
    from torchocr.losses.det_db_loss import DBLoss

    _seed()
    labels = _labels()
    preds = {'res': torch.rand(2, 3, 64, 64)}
    for lvl in ('p4', 'p3', 'p2'):
        preds[f'aux_maps_{lvl}'] = torch.rand(2, 3, 64, 64)

    weights = {'aux_weight_p4': 0.2, 'aux_weight_p3': 0.3, 'aux_weight_p2': 0.4}
    loss_fn = DBLoss(main_loss_type='DiceFocalLoss', focal_gamma=2.5, **weights)
    out = loss_fn(preds, labels)

    main = loss_fn({'res': preds['res']}, labels)['loss']
    expected = main + sum(w * out[f'loss_aux_maps_{k[-2:]}'] for k, w in weights.items())
    assert torch.allclose(out['loss'], expected)
    assert set(f'loss_aux_maps_{l}' for l in ('p4', 'p3', 'p2')) <= set(out)


def test_dbloss_default_path_unchanged_and_ignores_aux_without_weight():
    from torchocr.losses.det_db_loss import DBLoss
    from torchocr.losses.det_basic_loss import BalanceLoss

    _seed()
    loss_fn = DBLoss()  # DiceLoss + OHEM, aux weights 0
    assert isinstance(loss_fn.bce_loss, BalanceLoss)
    labels = _labels()
    preds = {'res': torch.rand(2, 3, 64, 64), 'aux_maps_p2': torch.rand(2, 3, 64, 64)}
    out = loss_fn(preds, labels)
    assert not any(k.startswith('loss_aux') for k in out)


@pytest.mark.parametrize('size', V6_SIZES)
def test_v6_model_train_step_backprops_into_aux_heads(size):
    from torchocr.losses import build_loss
    from torchocr.modeling.architectures import build_model

    cfg = _v6_cfg(size)
    _seed()
    model = build_model(copy.deepcopy(cfg['Architecture'])).train()
    loss_fn = build_loss(copy.deepcopy(cfg['Loss']))

    preds = model(torch.randn(2, 3, 160, 160))
    assert {'aux_maps_p4', 'aux_maps_p3', 'aux_maps_p2'} <= set(preds)
    assert preds['aux_maps_p4'].shape == preds['res'].shape

    loss = loss_fn(preds, _labels(2, 160, 160))
    assert torch.isfinite(loss['loss'])
    loss['loss'].backward()
    for lvl in ('p4', 'p3', 'p2'):
        g = getattr(model.head, f'aux_binarize_{lvl}').conv1.weight.grad
        assert g is not None and g.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 160, 160))
    assert set(out) == {'res'}


def test_head_fix_nan_zeroes_nan_in_training_only():
    from torchocr.modeling.heads.det_db_head import Head

    head = Head(32, fix_nan=True).train()
    x = torch.randn(1, 32, 8, 8)
    x[0, :, 2, 2] = float('nan')
    out = head(x)
    assert torch.isfinite(out).all()

    head_off = Head(32, fix_nan=False).train()
    head_off.load_state_dict(head.state_dict())
    assert torch.isnan(head_off(x)).any()


# ------------------------------------------------------------------------ EMA
def test_ema_threshold_decay_and_bias_correction():
    from torchocr.utils.ema import ModelEMA

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    ema = ModelEMA(model, decay=0.9997, ema_decay_type='threshold')
    w = model[0].weight.detach().clone()

    ema.update(model)  # step 0: decay = min(0.9997, 1/10) = 0.1
    assert ema._decay == pytest.approx(0.1)
    assert torch.allclose(ema.apply()['0.weight'], w, atol=1e-6)

    # 'normal' decay + constant weights => bias-corrected EMA equals the weights.
    ema_n = ModelEMA(model, decay=0.9, ema_decay_type='normal')
    for _ in range(5):
        ema_n.update(model)
    assert torch.allclose(ema_n.apply()['0.weight'], w, atol=1e-5)


def test_ema_tracks_moving_weights_and_copies_int_buffers():
    from torchocr.utils.ema import ModelEMA

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4)).train()
    ema = ModelEMA(model, decay=0.5, ema_decay_type='normal')
    shadow = torch.zeros_like(model[0].weight)
    for i in range(4):
        with torch.no_grad():
            model[0].weight.add_(1.0)
        model(torch.randn(2, 3, 4, 4))  # bumps num_batches_tracked
        ema.update(model)
        shadow = 0.5 * shadow + 0.5 * model[0].weight.detach()
    applied = ema.apply()
    assert torch.allclose(applied['0.weight'], shadow / (1 - 0.5 ** 4), atol=1e-5)
    assert applied['1.num_batches_tracked'].item() == 4
    assert applied['1.num_batches_tracked'].dtype == torch.int64


def test_ckpt_roundtrip_with_ema():
    from torchocr.utils.ckpt import save_ckpt, load_ckpt
    from torchocr.utils.ema import ModelEMA

    model = torch.nn.Conv2d(3, 4, 1)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    ema = ModelEMA(model, decay=0.9)
    ema.update(model)
    with tempfile.TemporaryDirectory() as d:
        cfg = {'Global': {'output_dir': d, 'distributed': False}}
        save_ckpt(model, cfg, opt, sched, 3, 10, {}, is_best=False,
                  ema=ema.state_dict_for_save(), train_state=model.state_dict())
        cfg['Global']['checkpoints'] = os.path.join(d, 'latest.pth')
        status = load_ckpt(torch.nn.Conv2d(3, 4, 1), cfg, None, None)
    assert status['epoch'] == 4
    restored = ModelEMA(model)
    restored.set_state_dict(status['ema'])
    assert restored.step == 1
    assert 'train_state' in status['ema']


# ----------------------------------------------------------------- data ops
def test_random_crop_outputs_fixed_size_and_valid_polys():
    from torchocr.data.imaug import RandomCrop

    _seed(3)
    op = RandomCrop(size=[640, 640], max_tries=50, keep_ratio=True)
    for _ in range(20):
        out = op(_det_sample(1200, 900))
        assert out['image'].shape == (640, 640, 3)
        assert out['polys'].dtype == np.float32
        assert len(out['polys']) == len(out['ignore_tags']) == len(out['texts'])
        if len(out['polys']):
            assert out['polys'].min() >= -1e-3 and out['polys'].max() <= 640 + 1e-3


def test_random_perspective_fit_output_keeps_polys_consistent():
    from torchocr.data.imaug import RandomPerspective

    _seed(1)
    op = RandomPerspective(prob=1.0, scale=0.0, shear=20.0, perspective=0.001, fit_output=True,
                           fill_value=[0, 0, 0])
    out = op(_det_sample())
    h, w = out['image'].shape[:2]
    assert len(out['polys']) == len(out['ignore_tags']) == len(out['texts'])
    for p in out['polys']:
        cx, cy = p.mean(axis=0)
        assert 0 < cx < w and 0 < cy < h


def test_color_jitter_works_on_numpy_and_matches_paddle_formulas():
    from torchocr.data.imaug import ColorJitter
    from torchocr.data.imaug.ColorJitter import adjust_brightness, adjust_contrast

    _seed()
    img = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    out = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.1)({'image': img.copy()})['image']
    assert out.shape == img.shape and out.dtype == np.uint8

    np.testing.assert_array_equal(adjust_brightness(img, 1.5), np.clip(img * 1.5, 0, 255).astype(np.uint8))
    np.testing.assert_array_equal(adjust_contrast(img, 2.0),
                                  np.clip((img.astype(np.float64) - 74) * 2.0 + 74, 0, 255).astype(np.uint8))


def test_iaa_augment_albumentations_backend():
    pytest.importorskip('albumentations')
    from torchocr.data.imaug import IaaAugment

    data = _det_sample(100, 200)
    polys = data['polys'].copy()
    op = IaaAugment(backend='albumentations', augmenter_args=[
        {'type': 'Fliplr', 'args': {'p': 1.0}},
        {'type': 'Resize', 'args': {'size': [2, 2]}},
    ])
    out = op(data)
    assert out['image'].shape == (200, 400, 3)
    np.testing.assert_allclose(out['polys'][..., 1], polys[..., 1] * 2, atol=1e-3)
    np.testing.assert_allclose(np.sort(out['polys'][..., 0], axis=1),
                               np.sort((200 - polys[..., 0]) * 2, axis=1), atol=2 + 1e-3)

    # v6 Affine args (p / fit_output) only exist in this backend; op must pickle
    # for spawn-based DataLoader workers.
    op6 = IaaAugment(backend='albumentations', augmenter_args=[
        {'type': 'Affine', 'args': {'p': 1.0, 'rotate': [-45, 45], 'fit_output': True}},
        {'type': 'Resize', 'args': {'size': [0.1, 2]}},
    ])
    out = pickle.loads(pickle.dumps(op6))(_det_sample())
    assert out['polys'].shape[1:] == (4, 2)


def test_iaa_augment_default_backend_is_imgaug():
    from torchocr.data.imaug import IaaAugment

    op = IaaAugment()
    assert op.backend == 'imgaug'
    with pytest.raises(ValueError):
        IaaAugment(backend='nope')


@pytest.mark.parametrize('epoch_source', ['data', 'kwarg'])
def test_shrink_map_epoch_ramp(epoch_source):
    from torchocr.data.imaug import MakeShrinkMap

    def area(op, data):
        return op(copy.deepcopy(data))['shrink_map'].sum()

    data = _det_sample()
    base = area(MakeShrinkMap(shrink_ratio=0.4, total_epoch=100), data)
    if epoch_source == 'data':
        later = area(MakeShrinkMap(shrink_ratio=0.4, total_epoch=100), dict(data, epoch=100))
    else:
        later = area(MakeShrinkMap(shrink_ratio=0.4, total_epoch=100, epoch=100), data)
    # shrink_ratio 0.4 -> 0.6 => less shrinkage => larger kernels
    assert later > base


@pytest.mark.parametrize('size', V6_SIZES)
def test_v6_train_pipeline_end_to_end(size):
    pytest.importorskip('albumentations')
    from torchocr.data.imaug import create_operators, transform

    cfg = _v6_cfg(size)
    ops_cfg = [t for t in cfg['Train']['dataset']['transforms']
               if list(t)[0] not in ('DecodeImage', 'DetLabelEncode', 'CopyPaste')]
    ops = create_operators(copy.deepcopy(ops_cfg), cfg['Global'])
    _seed(5)
    for _ in range(5):
        out = transform(_det_sample(), ops)
        assert out is not None
        img, tmap, tmask, smap, smask = out
        assert img.shape == (3, 640, 640)
        assert tmap.shape == tmask.shape == smap.shape == smask.shape == (640, 640)


# ------------------------------------------------------------------ RepLKPAN
def test_replkpan_structure_matches_paddle_param_names():
    from torchocr.modeling.necks import build_neck

    neck = build_neck({'name': 'RepLKPAN', 'in_channels': [128, 256, 512, 896],
                       'out_channels': 256, 'intracl': True})
    keys = set(neck.state_dict())
    for k in ['ins_conv.0.weight', 'inp_conv.3.dw.lk_origin.weight', 'inp_conv.3.dw.dil_conv_k5_2.weight',
              'inp_conv.3.dw.dil_bn_k3_4.running_var', 'inp_conv.0.pw.weight', 'inp_conv.0.bn.weight',
              'pan_head_conv.2.weight', 'pan_lat_conv.3.dw.origin_bn.weight', 'incl4.c_layer_7x7.weight']:
        assert k in keys, k
    assert len(neck.pan_head_conv) == 3
    assert neck.inp_conv[0].dw.kernel_size == 9


@pytest.mark.parametrize('intracl', [False, True])
def test_replkpan_rep_is_equivalent(intracl):
    from torchocr.modeling.necks.db_fpn import RepLKPAN

    _seed()
    neck = RepLKPAN([16, 24, 32, 48], out_channels=32, intracl=intracl)
    # non-trivial BN statistics so the fusion is actually exercised
    for m in neck.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    neck.eval()
    feats = [torch.randn(1, c, 64 // s, 64 // s) for c, s in zip([16, 24, 32, 48], [1, 2, 4, 8])]
    with torch.no_grad():
        ref = neck(feats)
        neck.rep()
        out = neck(feats)
    assert out.shape == (1, 32, 64, 64)
    assert torch.allclose(ref, out, atol=1e-4)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in neck.inp_conv.modules())
