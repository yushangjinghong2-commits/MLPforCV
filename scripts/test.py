import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES
from src.data import prepare_data
from src.metrics import confusion_matrix_np, per_class_accuracy
from src.serialization import load_checkpoint
from src.trainer import evaluate
from src.visualization import plot_confusion_matrix, plot_prediction_examples


def build_parser():
    parser = argparse.ArgumentParser(description="Test a saved Fashion-MNIST MLP checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data/fashion-mnist")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--report-dir", type=str, default="model/test_reports")
    parser.add_argument("--visualization-dir", type=str, default="visualizations/test")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()

    checkpoint_path = Path(args.checkpoint)
    model, metadata = load_checkpoint(checkpoint_path)
    data = prepare_data(Path(args.data_dir), seed=metadata["config"]["seed"])

    test_loss, test_acc = evaluate(
        model,
        data["X_test"],
        data["y_test"],
        weight_decay=metadata["config"]["weight_decay"],
    )
    y_pred = model.predict(data["X_test"])
    cm = confusion_matrix_np(data["y_test"], y_pred, num_classes=10)
    class_acc = per_class_accuracy(cm)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{checkpoint_path.stem}_test.json"
    report_payload = {
        "checkpoint": str(checkpoint_path),
        "run_name": metadata["run_name"],
        "config": metadata["config"],
        "best_val_acc": metadata["best_val_acc"],
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "per_class_accuracy": {
            CLASS_NAMES[i]: float(acc) for i, acc in enumerate(class_acc)
        },
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    vis_dir = Path(args.visualization_dir)
    plot_confusion_matrix(cm, save_path=vis_dir / f"{checkpoint_path.stem}_confusion_matrix.png")

    correct_idx = np.where(y_pred == data["y_test"])[0]
    wrong_idx = np.where(y_pred != data["y_test"])[0]
    plot_prediction_examples(
        data["X_test"],
        data["y_test"],
        y_pred,
        correct_idx,
        "Correctly Classified Samples",
        save_path=vis_dir / f"{checkpoint_path.stem}_correct.png",
        n=10,
        seed=args.seed,
    )
    plot_prediction_examples(
        data["X_test"],
        data["y_test"],
        y_pred,
        wrong_idx,
        "Misclassified Samples",
        save_path=vis_dir / f"{checkpoint_path.stem}_wrong.png",
        n=10,
        seed=args.seed + 1,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")
    print(f"Report saved to: {report_path}")
    print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
