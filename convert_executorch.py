import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse

import torch

from tools.convert_utils import load_config, load_torch_model
from tools.utility import update_rec_head_out_channels
from torchocr.postprocess import build_post_process

from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

from executorch.exir import to_edge_transform_and_lower
from executorch.devtools.backend_debug import get_delegation_info
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML model config")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth weights file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pte path (default: weights/<weights_basename>_<backend>.pte)",
    )
    parser.add_argument(
        "--dims",
        type=str,
        required=True,
        help="Input shape as comma-separated ints, e.g. 1,3,48,320",
    )
    parser.add_argument("--backend", type=str, default="coreml")
    return parser.parse_args()


def main():
    args = parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    output = args.output or (
        f"weights/{os.path.basename(args.weights).rsplit('.', 1)[0]}_{args.backend}.pte"
    )

    device = "cpu"
    config = load_config(args.config)
    post_process_class = build_post_process(config["PostProcess"], config["Global"])
    update_rec_head_out_channels(config, post_process_class)
    model_config = config["Architecture"]
    print(model_config)

    torch_model = load_torch_model(model_config, device, args.weights)

    sample_inputs = (torch.randn(*dims, dtype=torch.float32),)

    if args.backend == "xnnpack":
        partitioner = [XnnpackPartitioner()]
    elif args.backend == "coreml":
        partitioner = [CoreMLPartitioner(lower_full_graph=True)]
    elif args.backend == "vulkan":
        partitioner = [VulkanPartitioner()]
    else:
        raise NotImplementedError(f"Unknown backend: {args.backend}")

    et_program = to_edge_transform_and_lower(
        torch.export.export(torch_model, sample_inputs, dynamic_shapes=None),
        partitioner=partitioner,
    )

    with open(output, "wb") as file:
        et_program.to_executorch().write_to_file(file)

    graph_module = et_program.exported_program().graph_module
    delegation_info = get_delegation_info(graph_module)
    print(delegation_info.get_summary())
    df = delegation_info.get_operator_delegation_dataframe()
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))


def inference_executorch(weights: str, dims: list[int]):
    from executorch.runtime import Runtime
    from typing import List

    runtime = Runtime.get()
    input_tensor: torch.Tensor = torch.randn(*dims, dtype=torch.float32)
    program = runtime.load_program(weights)
    method = program.load_method("forward")
    output: List[torch.Tensor] = method.execute([input_tensor])
    print("Run successfully via executorch")
    print(output[0].shape)


if __name__ == "__main__":
    main()
    # inference_executorch("weights/...", [1, 3, 48, 320])
