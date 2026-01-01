import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse

import yaml
import torch

from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process

from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim
from executorch.devtools.backend_debug import get_delegation_info
from tabulate import tabulate


def load_config(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


def init_head(config):
    global_config = config["Global"]
    post_process_class = build_post_process(config["PostProcess"], global_config)
    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num
                    # update SARLoss params
                    if (
                        list(config["Loss"]["loss_config_list"][-1].keys())[0]
                        == "DistillationSARLoss"
                    ):
                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"][
                            "ignore_index"
                        ] = (char_num + 1)
                        out_channels_list["SARLabelDecode"] = char_num + 2
                    elif (
                        list(config["Loss"]["loss_config_list"][-1].keys())[0]
                        == "DistillationNRTRLoss"
                    ):
                        out_channels_list["NRTRLabelDecode"] = char_num + 3

                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
            # update SARLoss params
            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                        "ignore_index": char_num + 1
                    }
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                        char_num + 1
                    )
                out_channels_list["SARLabelDecode"] = char_num + 2
            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":
                out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

        if config["PostProcess"]["name"] == "SARLabelDecode":  # for SAR model
            config["Loss"]["ignore_index"] = char_num - 1
    return config


def load_torch_model(config, device, torch_pth: str) -> torch.nn.Module:
    print(f"torch version: {torch.__version__}")
    if device == "gpu":
        device = "cuda"
    model = build_model(config)
    model.load_state_dict(torch.load(torch_pth)["state_dict"])
    model.eval()
    model = model.to(device)

    return model


def comma_separated_list_int(arg):
    try:
        return [int(x) for x in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid list format: {arg}. Use comma-separated integers."
        )


class ExportConfig:
    def __init__(self, det: bool, backend: str = "coreml"):
        if det:
            self.config_path: str = "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml"
            self.weights_path: str = "weights/mob_det_2.pth"
            self.backend: str = backend
            self.dims: list[int] = [1, 3, 1280, 704]
            self.dynamic: dict = None
        else:
            self.config_path: str = "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
            self.weights_path: str = "weights/mob_rec_2.pth"
            self.backend: str = backend
            self.dims: list[int] = [1, 3, 48, 320]
            # self.dynamic: dict = {
            #     "x": {
            #         0: Dim("batchsize", max=4),
            #     }
            # }
            self.dynamic: dict = None
        self.output_path: str = (
            f"weights/{os.path.basename(self.weights_path).rsplit('.', 1)[0]}_{backend}.pte"
        )


def parse_export_config() -> ExportConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det", action="store_true", default=False)
    parser.add_argument("--backend", type=str, default="coreml")
    args = parser.parse_args()

    cfg = ExportConfig(args.det, args.backend)
    return cfg


def main(cfg: ExportConfig):
    device = "cpu"
    _config = load_config(cfg.config_path)
    _config = init_head(_config)
    model_config = _config["Architecture"]
    print(model_config)

    torch_pth = cfg.weights_path
    torch_model = load_torch_model(model_config, device, torch_pth)

    sample_inputs = (torch.randn(*cfg.dims, dtype=torch.float32),)

    dynamic_shapes = cfg.dynamic

    if cfg.backend == "xnnpack":
        partitioner = [XnnpackPartitioner()]
    elif cfg.backend == "coreml":
        partitioner = [CoreMLPartitioner(lower_full_graph=True)]
    elif cfg.backend == "vulkan":
        partitioner = [VulkanPartitioner()]
    else:
        raise NotImplementedError

    et_program = to_edge_transform_and_lower(
        torch.export.export(
            torch_model,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
        ),
        partitioner=partitioner,
    )

    with open(cfg.output_path, "wb") as file:
        et_program.to_executorch().write_to_file(file)

    # Get delegation info
    graph_module = et_program.exported_program().graph_module
    delegation_info = get_delegation_info(graph_module)
    # print the summary like the number of delegated nodes, non-delegated nodes, etc
    print(delegation_info.get_summary())
    df = delegation_info.get_operator_delegation_dataframe()
    # print the table including op_type, occurrences_in_delegated_graphs, occurrences_in_non_delegated_graphs
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))


def inference_executorch(cfg: ExportConfig):
    import torch
    from executorch.runtime import Runtime
    from typing import List

    runtime = Runtime.get()

    input_tensor: torch.Tensor = torch.randn(*cfg.dims, dtype=torch.float32)
    program = runtime.load_program(cfg.output_path)
    method = program.load_method("forward")
    output: List[torch.Tensor] = method.execute([input_tensor])
    print("Run successfully via executorch")
    print(output[0].shape)


if __name__ == "__main__":
    cfg = parse_export_config()
    main(cfg)
    # inference_executorch(cfg)
