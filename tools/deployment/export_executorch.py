import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

import torch

from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils.ckpt import load_ckpt
from torchocr.utils.logging import get_logger
from torchocr import Config
from tools.utility import update_rec_head_out_channels, ArgsParser
from tools.infer.pytorch.infer_rec import build_rec_process
from tools.infer.pytorch.infer_det import build_det_process
from tools.infer.pytorch.infer_cls import build_cls_process

from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

from executorch.exir import to_edge_transform_and_lower
from executorch.devtools.backend_debug import get_delegation_info
from tabulate import tabulate
import coremltools


# CPU goes through XNNPACK; Apple GPU/ANE through CoreML. "mps"/"metal" lowers to
# the Metal Performance Shaders delegate — note the prebuilt `executorch` pip
# wheel does NOT bundle the MPS runtime (only CoreML + XNNPACK), so an mps .pte
# exports fine but only runs against an executorch built with -DEXECUTORCH_BUILD_MPS=ON.
# On Apple Silicon prefer CoreML for GPU acceleration (see ExecuTorch backend docs).
def _make_partitioner(backend):
    if backend == "xnnpack":
        return [XnnpackPartitioner()]
    if backend == "coreml":
        # Force the CoreML deployment target. compile_specs must be a list of
        # CompileSpec (not a coremltools.target enum) — generate_compile_specs
        # builds them, defaulting to compute_unit=ALL, compute_precision=fp16.
        return [CoreMLPartitioner(
            lower_full_graph=True,
            compile_specs=CoreMLBackend.generate_compile_specs(
                minimum_deployment_target=coremltools.target.iOS16,
            ),
        )]
    if backend in ("mps", "metal"):
        raise NotImplementedError(f"MPS/Metal is deprecated. Use coreml instead.")
    if backend == "vulkan":
        return [VulkanPartitioner()]
    raise NotImplementedError(f"Unknown backend: {backend}")


def export_single_model(model, _cfg, export_dir, export_config, logger, backend):
    for layer in model.modules():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
            layer.rep()
    os.makedirs(export_dir, exist_ok=True)

    # Save the pre/post-processing config next to the .pte so the inference
    # runner (tools/infer/executorch/predict_det.py) is config-driven, exactly
    # like the ONNX/MNN exports.
    export_cfg = {"PostProcess": _cfg["PostProcess"]}
    if _cfg["Architecture"]["model_type"] == "rec":
        export_cfg["Transforms"] = build_rec_process(_cfg)
    elif _cfg["Architecture"]["model_type"] == "det":
        export_cfg["Transforms"] = build_det_process(_cfg)
    elif _cfg["Architecture"]["model_type"] == "cls":
        export_cfg["Transforms"] = build_cls_process(_cfg)
    cfg.save(os.path.join(export_dir, "config.yaml"), export_cfg)

    dummy_input = torch.randn(*export_config["export_shape"], device="cpu")
    save_path = os.path.join(export_dir, f"model_{backend}.pte")

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, (dummy_input,), dynamic_shapes=None),
        partitioner=_make_partitioner(backend),
    )

    with open(save_path, "wb") as f:
        et_program.to_executorch().write_to_file(f)

    graph_module = et_program.exported_program().graph_module
    delegation_info = get_delegation_info(graph_module)
    logger.info(delegation_info.get_summary())
    df = delegation_info.get_operator_delegation_dataframe()
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
    logger.info(f"finish export model to {save_path}")


def main(cfg, backend):
    _cfg = cfg.cfg
    logger = get_logger()
    global_config = _cfg["Global"]
    export_config = _cfg["Export"]

    post_process_class = build_post_process(_cfg["PostProcess"])
    update_rec_head_out_channels(_cfg, post_process_class)
    model = build_model(_cfg["Architecture"])

    load_ckpt(model, _cfg)
    model.eval()

    export_dir = export_config.get("export_dir", "")
    if not export_dir:
        export_dir = os.path.join(global_config.get("output_dir", "output"), "export")

    if _cfg["Architecture"]["algorithm"] in ["Distillation"]:
        _cfg["PostProcess"]["name"] = post_process_class.__class__.__base__.__name__
        for model_name in model.model_list:
            sub_dir = os.path.join(export_dir, model_name)
            export_single_model(
                model.model_list[model_name], _cfg, sub_dir, export_config, logger, backend
            )
    else:
        export_single_model(model, _cfg, export_dir, export_config, logger, backend)


def parse_args():
    parser = ArgsParser()
    parser.add_argument("--backend", type=str, default="coreml",
                        help="executorch backend: coreml | xnnpack | vulkan")
    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop("opt")
    cfg.merge_dict(opt)
    main(cfg, FLAGS["backend"])
