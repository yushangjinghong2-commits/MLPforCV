import json
from pathlib import Path

import numpy as np

from .models import MLP


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_checkpoint(path: Path, model: MLP, metadata: dict) -> None:
    payload = {
        "metadata_json": json.dumps(metadata, default=_json_default),
        "W1": model.params["W1"],
        "b1": model.params["b1"],
        "W2": model.params["W2"],
        "b2": model.params["b2"],
    }
    np.savez(path, **payload)


def load_checkpoint(path: Path):
    checkpoint = np.load(path, allow_pickle=False)
    metadata = json.loads(str(checkpoint["metadata_json"]))
    config = metadata["config"]
    model = MLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        activation=config["activation"],
        seed=config["seed"],
    )
    model.params["W1"] = checkpoint["W1"]
    model.params["b1"] = checkpoint["b1"]
    model.params["W2"] = checkpoint["W2"]
    model.params["b2"] = checkpoint["b2"]
    return model, metadata
