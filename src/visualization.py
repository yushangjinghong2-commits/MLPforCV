import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import CLASS_NAMES


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_history(history, title, save_path: Path):
    ensure_parent(save_path)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, save_path: Path, class_names=None):
    ensure_parent(save_path)
    class_names = class_names or CLASS_NAMES
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() * 0.5 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_examples(X, y_true, y_pred, indices, title, save_path: Path, n=10, seed=42):
    ensure_parent(save_path)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
    cols = 5
    rows = max(1, math.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, chosen):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"T:{CLASS_NAMES[y_true[idx]]}\nP:{CLASS_NAMES[y_pred[idx]]}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
