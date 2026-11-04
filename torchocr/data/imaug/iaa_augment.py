import numpy as np

# imgaug uses np.sctypes which was removed in NumPy 2.0
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'int': [np.int8, np.int16, np.int32, np.int64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }

import imgaug
import imgaug.augmenters as iaa


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(
                    *[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{
                k: self.to_tuple_if_list(v)
                for k, v in args['args'].items()
            })
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list) and len(obj) == 2:
            return tuple(obj)
        return obj


# ---------------------------------------------------------------------------
# albumentations backend -- ported verbatim from PaddleOCR
# ppocr/data/imaug/iaa_augment.py (the version PP-OCRv6 trains with).
# Optional dependency: only needed when IaaAugment(backend='albumentations').
# ---------------------------------------------------------------------------
import os

# Prevent automatic updates in Albumentations for stability in augmentation behavior
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

try:
    import albumentations as A
    from albumentations.core.transforms_interface import DualTransform
    from albumentations.augmentations.geometric import functional as fgeometric
    from packaging import version

    ALBU_VERSION = version.parse(A.__version__)
    IS_ALBU_NEW_VERSION = ALBU_VERSION >= version.parse("1.4.15")
    _HAS_ALBU = True
except ImportError:  # pragma: no cover
    _HAS_ALBU = False

if _HAS_ALBU:
    # Custom resize transformation mimicking Imgaug's behavior with scaling
    class ImgaugLikeResize(DualTransform):
        def __init__(self, scale_range=(0.5, 3.0), interpolation=1, p=1.0):
            super(ImgaugLikeResize, self).__init__(p)
            self.scale_range = scale_range
            self.interpolation = interpolation

        # Resize the image based on a randomly chosen scale within the scale range
        def apply(self, img, scale=1.0, **params):
            height, width = img.shape[:2]
            new_height = int(height * scale)
            new_width = int(width * scale)

            if IS_ALBU_NEW_VERSION:
                return fgeometric.resize(
                    img, (new_height, new_width), interpolation=self.interpolation
                )
            return fgeometric.resize(
                img, new_height, new_width, interpolation=self.interpolation
            )

        # Apply the same scaling transformation to keypoints (e.g., polygon points)
        def apply_to_keypoints(self, keypoints, scale=1.0, **params):
            return np.array(
                [(x * scale, y * scale) + tuple(rest) for x, y, *rest in keypoints]
            )

        # Get random scale parameter within the specified range
        def get_params(self):
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            return {"scale": scale}


# Builder class to translate custom augmenter arguments into Albumentations-compatible format
class AlbuAugmenterBuilder(object):
    def __init__(self):
        # Map common Imgaug transformations to equivalent Albumentations transforms
        self.imgaug_to_albu = {
            "Fliplr": "HorizontalFlip",
            "Flipud": "VerticalFlip",
            "Affine": "Affine",
            # Additional mappings can be added here if needed
        }

    # Recursive method to construct augmentation pipeline based on provided arguments
    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            # Build the full augmentation sequence if it's a root-level call
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return A.Compose(
                    sequence,
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False
                    ),
                )
            else:
                # Build individual augmenters for nested arguments
                augmenter_type = args[0]
                augmenter_args = args[1] if len(args) > 1 else {}
                augmenter_args_mapped = self.map_arguments(
                    augmenter_type, augmenter_args
                )
                augmenter_type_mapped = self.imgaug_to_albu.get(
                    augmenter_type, augmenter_type
                )
                if augmenter_type_mapped == "Resize":
                    return ImgaugLikeResize(**augmenter_args_mapped)
                else:
                    cls = getattr(A, augmenter_type_mapped)
                    return cls(
                        **{
                            k: self.to_tuple_if_list(v)
                            for k, v in augmenter_args_mapped.items()
                        }
                    )
        elif isinstance(args, dict):
            # Process individual transformation specified as dictionary
            augmenter_type = args["type"]
            augmenter_args = args.get("args", {})
            augmenter_args_mapped = self.map_arguments(augmenter_type, augmenter_args)
            augmenter_type_mapped = self.imgaug_to_albu.get(
                augmenter_type, augmenter_type
            )
            if augmenter_type_mapped == "Resize":
                return ImgaugLikeResize(**augmenter_args_mapped)
            else:
                cls = getattr(A, augmenter_type_mapped)
                return cls(
                    **{
                        k: self.to_tuple_if_list(v)
                        for k, v in augmenter_args_mapped.items()
                    }
                )
        else:
            raise RuntimeError("Unknown augmenter arg: " + str(args))

    # Map arguments to expected format for each augmenter type
    def map_arguments(self, augmenter_type, augmenter_args):
        augmenter_args = augmenter_args.copy()  # Avoid modifying the original arguments
        if augmenter_type == "Resize":
            # Ensure size is a valid 2-element list or tuple
            size = augmenter_args.get("size")
            if size:
                if not isinstance(size, (list, tuple)) or len(size) != 2:
                    raise ValueError(
                        f"'size' must be a list or tuple of two numbers, but got {size}"
                    )
                min_scale, max_scale = size
                return {
                    "scale_range": (min_scale, max_scale),
                    "interpolation": 1,  # Linear interpolation
                    "p": 1.0,
                }
            else:
                return {"scale_range": (1.0, 1.0), "interpolation": 1, "p": 1.0}
        elif augmenter_type == "Affine":
            # Map rotation to a tuple and ensure p=1.0 to apply transformation
            rotate = augmenter_args.get("rotate", 0)
            if isinstance(rotate, list):
                rotate = tuple(rotate)
            elif isinstance(rotate, (int, float)):
                rotate = (float(rotate), float(rotate))
            augmenter_args["rotate"] = rotate
            augmenter_args["p"] = augmenter_args.get("p", 1.0)
            if augmenter_args.get("fit_output") is True:
                if "border_mode" not in augmenter_args:
                    import cv2

                    augmenter_args["border_mode"] = cv2.BORDER_CONSTANT
                if "fill" not in augmenter_args:
                    augmenter_args["fill"] = 0
            return augmenter_args
        else:
            # For other augmenters, ensure 'p' probability is specified
            p = augmenter_args.get("p", 1.0)
            augmenter_args["p"] = p
            return augmenter_args

    # Convert lists to tuples for Albumentations compatibility
    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment():
    """Image + polygon geometric augmentation.

    backend:
      * 'imgaug' (default): original torchocr behaviour.
      * 'albumentations': PaddleOCR's current implementation, used by the
        PP-OCRv6 configs (supports Affine ``p`` / ``fit_output``).
    """

    def __init__(self, augmenter_args=None, backend='imgaug', **kwargs):
        if augmenter_args is None:
            augmenter_args = [{
                'type': 'Fliplr',
                'args': {
                    'p': 0.5
                }
            }, {
                'type': 'Affine',
                'args': {
                    'rotate': [-10, 10]
                }
            }, {
                'type': 'Resize',
                'args': {
                    'size': [0.5, 3]
                }
            }]
        if backend not in ('imgaug', 'albumentations'):
            raise ValueError(f"IaaAugment backend must be 'imgaug' or 'albumentations', got {backend!r}")
        self.backend = backend
        if backend == 'albumentations':
            if not _HAS_ALBU:
                raise ImportError("IaaAugment(backend='albumentations') requires `pip install albumentations`")
            self.augmenter = AlbuAugmenterBuilder().build(augmenter_args)
        else:
            self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        if self.backend == 'albumentations':
            return self._call_albu(data)
        image = data['image']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(
                keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

    # albumentations backend (PaddleOCR implementation)
    def _call_albu(self, data):
        image = data["image"]

        if self.augmenter:
            # Flatten polygons to individual keypoints for transformation
            keypoints = []
            keypoints_lengths = []
            for poly in data["polys"]:
                keypoints.extend([tuple(point) for point in poly])
                keypoints_lengths.append(len(poly))

            # Apply the augmentation pipeline to image and keypoints
            transformed = self.augmenter(image=image, keypoints=keypoints)
            data["image"] = transformed["image"]

            # Extract transformed keypoints and reconstruct polygon structures
            transformed_keypoints = transformed["keypoints"]

            # Reassemble polygons from transformed keypoints
            new_polys = []
            idx = 0
            for length in keypoints_lengths:
                new_poly = transformed_keypoints[idx : idx + length]
                new_polys.append(np.array([kp[:2] for kp in new_poly]))
                idx += length
            data["polys"] = np.array(new_polys)
        return data
