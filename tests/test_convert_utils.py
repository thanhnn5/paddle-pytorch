import os
import pytest
import tempfile
import yaml


def test_load_config_valid_yaml():
    from tools.convert_utils import load_config

    content = {"Global": {"device": "cpu"}, "Architecture": {}}
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        yaml.dump(content, f)
        path = f.name

    try:
        cfg = load_config(path)
        assert cfg["Global"]["device"] == "cpu"
    finally:
        os.unlink(path)


def test_load_config_rejects_non_yaml():
    from tools.convert_utils import load_config

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        with pytest.raises(AssertionError):
            load_config(path)
    finally:
        os.unlink(path)
