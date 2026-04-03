import os

import yaml


def load_config(file_path: str) -> dict:
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


def load_torch_model(config: dict, device: str, torch_pth: str):
    import torch
    from torchocr.modeling.architectures import build_model

    print(f"torch version: {torch.__version__}")
    if device == "gpu":
        device = "cuda"
    model = build_model(config)
    model.load_state_dict(torch.load(torch_pth)["state_dict"])
    model.eval()
    model = model.to(device)
    return model
