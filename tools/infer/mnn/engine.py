import os

import numpy as np
import MNN
from MNN import nn, expr


# MNN forward types — values must match the MNNForwardType enum in
# include/MNN/MNNForwardType.h. Requesting an id that the linked libMNN was
# not built with (e.g. 'coreml' on Linux, 'cuda' on macOS) makes MNN silently
# fall back to its backupType (CPU) and print "Can't Find type=N backend".
BACKEND_MAP = {
    'cpu':    0,   # MNN_FORWARD_CPU
    'metal':  1,   # MNN_FORWARD_METAL    — Apple Metal GPU (macOS/iOS)
    'cuda':   2,   # MNN_FORWARD_CUDA
    'opencl': 3,   # MNN_FORWARD_OPENCL
    'auto':   4,   # MNN_FORWARD_AUTO     — MNN picks
    'coreml': 5,   # MNN_FORWARD_NN       — CoreML registers itself here
    'opengl': 6,   # MNN_FORWARD_OPENGL
    'vulkan': 7,   # MNN_FORWARD_VULKAN
    # NNAPI / user-slot backends are only valid when libMNN is built with the
    # matching option and registers itself into MNN_FORWARD_USER_{0..3}.
    # Don't add an alias unless the build registers it; otherwise it falls
    # back to CPU silently.
}

# BackendConfig::PrecisionMode — see include/MNN/Interpreter.hpp.
# Note: 'normal' is fp16-where-the-backend-allows, NOT fp32. 'high' is the
# strict fp32 path.
PRECISION_MAP = {
    'normal': 0,   # Precision_Normal  — fp16 storage/compute where supported
    'high':   1,   # Precision_High    — strict fp32
    'low':    2,   # Precision_Low     — int8/quantised fast path where supported
    'low_bf': 3,   # Precision_Low_BF16
}


class MNNEngine:
    def __init__(self, mnn_path, backend='metal', precision='low', input_names=None, output_names=None):
        if not os.path.exists(mnn_path):
            raise Exception(f'{mnn_path} is not exists')

        if backend not in BACKEND_MAP:
            raise ValueError(f'unknown backend {backend!r}, choose from {list(BACKEND_MAP)}')
        if precision not in PRECISION_MAP:
            raise ValueError(f'unknown precision {precision!r}, choose from {list(PRECISION_MAP)}')

        config = {
            'backend': BACKEND_MAP[backend],
            'precision': precision,
            'numThread': 1,
        }
        self.runtime_manager = nn.create_runtime_manager((config,))
        self.input_names = input_names or ['input']
        self.output_names = output_names or ['output']
        self.net = nn.load_module_from_file(
            mnn_path,
            self.input_names,
            self.output_names,
            runtime_manager=self.runtime_manager,
        )

    def run(self, image_numpy):
        image_numpy = np.ascontiguousarray(image_numpy.astype(np.float32))
        input_var = expr.const(
            image_numpy.tobytes(),
            list(image_numpy.shape),
            expr.NCHW,
            expr.float,
        )
        outputs = self.net.forward([input_var])
        results = []
        for out in outputs:
            out = expr.convert(out, expr.NCHW)
            results.append(np.array(out.read(), copy=True))
        return results
