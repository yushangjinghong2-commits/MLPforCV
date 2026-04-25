import gzip
import urllib.request
from pathlib import Path

import numpy as np

from .constants import BASE_URL, FILES


def download_file(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.exists():
        url = BASE_URL + filename
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28 * 28).astype(np.float32)


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data.astype(np.int64)


def train_val_split(X, y, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    val_size = int(len(X) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def prepare_data(data_dir: Path, seed: int, val_ratio: float = 0.1):
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: download_file(data_dir, v) for k, v in FILES.items()}

    X_train_full = load_images(paths["train_images"])
    y_train_full = load_labels(paths["train_labels"])
    X_test = load_images(paths["test_images"])
    y_test = load_labels(paths["test_labels"])

    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0

    pixel_mean = X_train_full.mean(axis=0, keepdims=True)
    pixel_std = X_train_full.std(axis=0, keepdims=True) + 1e-8

    X_train_full = (X_train_full - pixel_mean) / pixel_std
    X_test = (X_test - pixel_mean) / pixel_std

    X_train, y_train, X_val, y_val = train_val_split(
        X_train_full, y_train_full, val_ratio=val_ratio, seed=seed
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "pixel_mean": pixel_mean.astype(np.float32),
        "pixel_std": pixel_std.astype(np.float32),
    }
