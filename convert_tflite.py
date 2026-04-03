import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse

import torch

from tools.convert_utils import load_config, load_torch_model
from tools.utility import update_rec_head_out_channels
from torchocr.postprocess import build_post_process

import litert_torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML model config")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth weights file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .tflite path (default: weights/<weights_basename>_<backend>.tflite)",
    )
    parser.add_argument(
        "--dims",
        type=str,
        required=True,
        help="Input shape as comma-separated ints, e.g. 1,3,48,320",
    )
    parser.add_argument("--swap_channel", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    output = args.output or (
        f"weights/{os.path.basename(args.weights).rsplit('.', 1)[0]}.tflite"
    )

    device = "cpu"
    config = load_config(args.config)
    post_process_class = build_post_process(config["PostProcess"], config["Global"])
    update_rec_head_out_channels(config, post_process_class)
    model_config = config["Architecture"]
    print(model_config)

    torch_model = load_torch_model(model_config, device, args.weights)

    sample_inputs = (torch.randn(*dims, dtype=torch.float32),)
    if args.swap_channel:
        torch_channel_last_model = litert_torch.to_channel_last_io(torch_model, args=[0])
        edge_model = litert_torch.convert(torch_channel_last_model.eval(), sample_inputs)
    else:
        edge_model = litert_torch.convert(torch_model.eval(), sample_inputs)
    edge_model.export(output)


if __name__ == "__main__":
    main()
