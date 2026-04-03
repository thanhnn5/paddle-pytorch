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

    # Verify converted model output matches PyTorch model
    x = sample_inputs[0]
    with torch.no_grad():
        if args.swap_channel:
            # PyTorch model expects channel-first; x is channel-last (e.g. 1,H,W,3)
            torch_out = torch_model(x.permute(0, 3, 1, 2))
        else:
            torch_out = torch_model(x)

    # Flatten dict/tuple outputs to a single tensor for comparison
    if isinstance(torch_out, dict):
        torch_out = next(iter(torch_out.values()))
    if isinstance(torch_out, (tuple, list)):
        torch_out = torch_out[0]

    tflite_out = edge_model(x.numpy())
    if isinstance(tflite_out, dict):
        tflite_out = next(iter(tflite_out.values()))
    if isinstance(tflite_out, (tuple, list)):
        tflite_out = tflite_out[0]
    tflite_tensor = torch.from_numpy(tflite_out)

    print("Torchout: ", torch_out.shape)
    print("Tfliteout: ", tflite_tensor.shape)

    # FP16 backends (XNNPACK, CoreML) accumulate ~1e-2 error per op — use a
    # realistic tolerance rather than FP32 epsilon.
    atol = 5e-2
    diff = (torch_out - tflite_tensor).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pct_above_atol = (diff > atol).float().mean().item() * 100
    match = max_diff < atol
    print(f"Verification {'PASSED' if match else 'FAILED'} (atol={atol})")
    print(f"  max  abs diff : {max_diff:.6f}")
    print(f"  mean abs diff : {mean_diff:.6f}")
    print(f"  % pixels > {atol}: {pct_above_atol:.2f}%")


if __name__ == "__main__":
    main()
