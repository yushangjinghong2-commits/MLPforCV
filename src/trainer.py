import random
from pathlib import Path

import numpy as np

from .serialization import save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def iterate_minibatches(X, y, batch_size, shuffle=True, seed=None):
    indices = np.arange(len(X))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def evaluate(model, X, y, batch_size=1024, weight_decay=0.0):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for xb, yb in iterate_minibatches(X, y, batch_size, shuffle=False):
        loss, _, logits = model.loss_and_grads(xb, yb, weight_decay=weight_decay)
        total_loss += loss * len(xb)
        total_correct += np.sum(np.argmax(logits, axis=1) == yb)
        total_samples += len(xb)
    return total_loss / total_samples, total_correct / total_samples


def clone_params(params):
    return {k: v.copy() for k, v in params.items()}


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    run_name,
    checkpoint_dir: Path,
    normalization_stats,
    epochs=15,
    batch_size=128,
    lr=0.1,
    lr_decay=0.95,
    weight_decay=1e-4,
    seed=42,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_checkpoint_path = None
    current_lr = lr

    for epoch in range(1, epochs + 1):
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, shuffle=True, seed=seed + epoch):
            _, grads, _ = model.loss_and_grads(xb, yb, weight_decay=weight_decay)
            model.update(grads, current_lr)

        train_loss, train_acc = evaluate(model, X_train, y_train, weight_decay=weight_decay)
        val_loss, val_acc = evaluate(model, X_val, y_val, weight_decay=weight_decay)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"lr={current_lr:.5f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = checkpoint_dir / f"{run_name}_best.npz"
            save_checkpoint(
                best_checkpoint_path,
                model,
                {
                    "run_name": run_name,
                    "epoch": epoch,
                    "best_val_acc": val_acc,
                    "history": history,
                    "normalization_stats": normalization_stats,
                    "config": {
                        "hidden_dim": model.hidden_dim,
                        "activation": model.activation,
                        "input_dim": model.input_dim,
                        "output_dim": model.output_dim,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "lr_decay": lr_decay,
                        "weight_decay": weight_decay,
                        "seed": seed,
                    },
                },
            )

        current_lr *= lr_decay

    return {
        "best_val_acc": best_val_acc,
        "history": history,
        "best_checkpoint_path": best_checkpoint_path,
    }
