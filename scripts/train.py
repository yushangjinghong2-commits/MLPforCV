import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import prepare_data
from src.models import MLP
from src.trainer import set_seed, train_model
from src.visualization import plot_history


def json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on Fashion-MNIST.")
    parser.add_argument("--data-dir", type=str, default="data/fashion-mnist")
    parser.add_argument("--checkpoint-dir", type=str, default="model/weights")
    parser.add_argument("--metrics-dir", type=str, default="model/train_metrics")
    parser.add_argument("--visualization-dir", type=str, default="visualizations/train")
    parser.add_argument("--run-name", type=str, default="mlp_run")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    data = prepare_data(Path(args.data_dir), seed=args.seed)
    model = MLP(784, args.hidden_dim, 10, activation=args.activation, seed=args.seed)

    result = train_model(
        model,
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        run_name=args.run_name,
        checkpoint_dir=Path(args.checkpoint_dir),
        normalization_stats={"pixel_mean": data["pixel_mean"], "pixel_std": data["pixel_std"]},
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{args.run_name}.json"
    metrics_payload = {
        "run_name": args.run_name,
        "hidden_dim": args.hidden_dim,
        "activation": args.activation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_val_acc": result["best_val_acc"],
        "best_checkpoint_path": str(result["best_checkpoint_path"]) if result["best_checkpoint_path"] else None,
        "history": result["history"],
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, default=json_default),
        encoding="utf-8",
    )

    vis_path = Path(args.visualization_dir) / f"{args.run_name}_history.png"
    title = f"hidden={args.hidden_dim}, act={args.activation}"
    plot_history(result["history"], title=title, save_path=vis_path)

    print(f"Best checkpoint: {result['best_checkpoint_path']}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Visualization saved to: {vis_path}")


if __name__ == "__main__":
    main()
