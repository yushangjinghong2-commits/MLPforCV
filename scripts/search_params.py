import argparse
import csv
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import prepare_data
from src.models import MLP
from src.trainer import set_seed, train_model


def parse_int_list(raw: str):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Run hyper-parameter search for Fashion-MNIST MLP.")
    parser.add_argument("--data-dir", type=str, default="data/fashion-mnist")
    parser.add_argument("--checkpoint-dir", type=str, default="model/weights")
    parser.add_argument("--results-dir", type=str, default="model/search_results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dims", type=str, default="128,256,512,768")
    parser.add_argument("--activations", type=str, default="relu,tanh,sigmoid")
    parser.add_argument("--lrs", type=str, default="0.2,0.1,0.05,0.01")
    parser.add_argument("--lr-decays", type=str, default="1.0,0.98,0.95,0.9")
    parser.add_argument("--weight-decays", type=str, default="0.0,1e-5,1e-4,1e-3")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    hidden_dims = parse_int_list(args.hidden_dims)
    activations = parse_str_list(args.activations)
    lrs = parse_float_list(args.lrs)
    lr_decays = parse_float_list(args.lr_decays)
    weight_decays = parse_float_list(args.weight_decays)

    search_space = [
        {
            "hidden_dim": hidden_dim,
            "activation": activation,
            "lr": lr,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
        }
        for hidden_dim, activation, lr, lr_decay, weight_decay in product(
            hidden_dims, activations, lrs, lr_decays, weight_decays
        )
    ]

    data = prepare_data(Path(args.data_dir), seed=args.seed)
    normalization_stats = {"pixel_mean": data["pixel_mean"], "pixel_std": data["pixel_std"]}
    results = []

    print(f"Total search configs: {len(search_space)}")

    for config in search_space:
        run_name = (
            f"hd{config['hidden_dim']}_"
            f"{config['activation']}_"
            f"lr{config['lr']}_"
            f"decay{config['lr_decay']}_"
            f"wd{config['weight_decay']}"
        ).replace(".", "p")
        print("\n" + "=" * 80)
        print(f"Running search config: {config}")
        model = MLP(784, config["hidden_dim"], 10, activation=config["activation"], seed=args.seed)
        result = train_model(
            model,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            run_name=run_name,
            checkpoint_dir=Path(args.checkpoint_dir),
            normalization_stats=normalization_stats,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=config["lr"],
            lr_decay=config["lr_decay"],
            weight_decay=config["weight_decay"],
            seed=args.seed,
        )
        results.append(
            {
                "run_name": run_name,
                "hidden_dim": config["hidden_dim"],
                "activation": config["activation"],
                "lr": config["lr"],
                "lr_decay": config["lr_decay"],
                "weight_decay": config["weight_decay"],
                "epochs": args.epochs,
                "best_val_acc": result["best_val_acc"],
                "checkpoint_path": str(result["best_checkpoint_path"]) if result["best_checkpoint_path"] else "",
            }
        )

    results.sort(key=lambda item: item["best_val_acc"], reverse=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "search_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "hidden_dim",
                "activation",
                "lr",
                "lr_decay",
                "weight_decay",
                "epochs",
                "best_val_acc",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSearch results:")
    for row in results:
        print(row)
    print(f"\nSaved search report to: {csv_path}")


if __name__ == "__main__":
    main()
