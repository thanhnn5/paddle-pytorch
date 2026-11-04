"""ColorJitter matching PaddleOCR's ``ppocr/data/imaug/ColorJitter.py``.

Paddle wraps ``paddle.vision.transforms.ColorJitter``, which on numpy (HWC,
BGR, uint8) images dispatches to its cv2 backend. torchvision's ColorJitter
only accepts PIL/Tensor and raises on numpy, so the cv2 backend is
re-implemented here (paddle/vision/transforms/functional_cv2.py), including
its quirks:
  * contrast pivots around 74 (not the image mean);
  * saturation re-samples its own blend factor inside the sampled range;
  * the 4 sub-transforms are applied in a random order each call.
PIL / Tensor inputs fall back to torchvision.
"""
import random

import cv2
import numpy as np
from torchvision.transforms import ColorJitter as _TVColorJitter

__all__ = ['ColorJitter']


def _check_input(value, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError('ColorJitter value must be non negative.')
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f'ColorJitter values should be between {bound}')
        value = list(value)
    else:
        raise TypeError('ColorJitter value should be a number or a list/tuple of length 2.')
    if value[0] == value[1] == center:
        return None
    return value


def _lut(img, table):
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    return cv2.LUT(img, table)


def adjust_brightness(img, factor):
    table = np.array([i * factor for i in range(256)]).clip(0, 255).astype('uint8')
    return _lut(img, table)


def adjust_contrast(img, factor):
    table = np.array([(i - 74) * factor + 74 for i in range(256)]).clip(0, 255).astype('uint8')
    return _lut(img, table)


def adjust_saturation(img, factor):
    dtype = img.dtype
    img = img.astype(np.float32)
    alpha = np.random.uniform(max(0, 1 - factor), 1 + factor)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    img = img * alpha + gray * (1 - alpha)
    return img.clip(0, 255).astype(dtype)


def adjust_hue(img, factor):
    if not (-0.5 <= factor <= 0.5):
        raise ValueError(f'hue_factor:{factor} is not in [-0.5, 0.5].')
    dtype = img.dtype
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv)
    h = ((h.astype(np.int32) + int(factor * 255)) % 256).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR_FULL).astype(dtype)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, **kwargs):
        self.brightness = _check_input(brightness)
        self.contrast = _check_input(contrast)
        self.saturation = _check_input(saturation)
        self.hue = _check_input(hue, center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self._tv = _TVColorJitter(brightness, contrast, saturation, hue)

    def _jitter_np(self, img):
        ops = []
        if self.brightness is not None:
            ops.append((adjust_brightness, self.brightness))
        if self.contrast is not None:
            ops.append((adjust_contrast, self.contrast))
        if self.saturation is not None:
            ops.append((adjust_saturation, self.saturation))
        if self.hue is not None:
            ops.append((adjust_hue, self.hue))
        random.shuffle(ops)
        for fn, (lo, hi) in ops:
            img = fn(img, random.uniform(lo, hi))
        return img

    def __call__(self, data):
        image = data['image']
        if isinstance(image, np.ndarray):
            image = self._jitter_np(image)
        else:
            image = self._tv(image)
        data['image'] = image
        return data
