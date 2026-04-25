import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serialization import load_checkpoint


def normalize_kernel(kernel):
    kernel = kernel.reshape(28, 28)
    min_val = kernel.min()
    max_val = kernel.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(kernel)
    return (kernel - min_val) / (max_val - min_val)


def plot_first_layer_weights(W1, save_path: Path, max_filters=128):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(W1.shape[1], max_filters)
    cols = 8
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(-1)

    for idx in range(count):
        axes[idx].imshow(normalize_kernel(W1[:, idx]), cmap="gray")
        axes[idx].axis("off")

    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle("First-Layer Hidden Weights", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize first-layer weights from a saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="visualizations/train/first_layer_weights.png")
    parser.add_argument("--max-filters", type=int, default=128)
    args = parser.parse_args()

    model, _ = load_checkpoint(Path(args.checkpoint))
    plot_first_layer_weights(model.params["W1"], Path(args.output), max_filters=args.max_filters)
    print(f"Saved weight visualization to: {args.output}")


if __name__ == "__main__":
    main()
