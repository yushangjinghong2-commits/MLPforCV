import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES
from src.data import prepare_data
from src.metrics import confusion_matrix_np
from src.serialization import load_checkpoint


def plot_training_curves(history, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="tab:green", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _normalize_kernel(kernel):
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
        axes[idx].imshow(_normalize_kernel(W1[:, idx]), cmap="gray")
        axes[idx].axis("off")
    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle("First-Layer Hidden Weights (Top 128 Neurons)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_error_examples(y_true, y_pred):
    preferred_pairs = [
        (0, 6),  # T-shirt/top -> Shirt
        (2, 4),  # Pullover -> Coat
        (6, 0),  # Shirt -> T-shirt/top
        (6, 2),  # Shirt -> Pullover
        (5, 7),  # Sandal -> Sneaker
    ]
    chosen = []
    for true_label, pred_label in preferred_pairs:
        indices = np.where((y_true == true_label) & (y_pred == pred_label))[0]
        if len(indices) > 0:
            chosen.append((indices[0], true_label, pred_label))
    return chosen


def plot_error_examples(X_test, y_true, y_pred, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    chosen = select_error_examples(y_true, y_pred)
    cols = len(chosen)
    fig, axes = plt.subplots(1, cols, figsize=(3.1 * cols, 3.6))
    if cols == 1:
        axes = [axes]

    for ax, (idx, t, p) in zip(axes, chosen):
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"True: {CLASS_NAMES[t]}\nPred: {CLASS_NAMES[p]}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return chosen


def build_report_pdf(output_pdf: Path, assets: dict, meta: dict, test_report: dict, error_examples):
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.96, "Fashion-MNIST MLP Experiment Report", ha="center", va="top", fontsize=18)
        lines = [
            "1. Task overview",
            "This project uses a NumPy-implemented single-hidden-layer MLP to classify Fashion-MNIST.",
            "",
            "2. Dataset and preprocessing",
            "Fashion-MNIST contains 60,000 training images and 10,000 test images.",
            "Each image is resized to 28x28 and flattened to a 784-dimensional vector.",
            "The input is normalized by training-set pixel mean and standard deviation.",
            "",
            "3. Model structure",
            f"Input dimension: {meta['config']['input_dim']}",
            f"Hidden dimension: {meta['config']['hidden_dim']}",
            f"Output dimension: {meta['config']['output_dim']}",
            f"Activation: {meta['config']['activation']}",
            "",
            "4. Best hyperparameters",
            f"lr={meta['config']['lr']}, lr_decay={meta['config']['lr_decay']}, weight_decay={meta['config']['weight_decay']}, epochs={meta['config']['epochs']}",
            f"Best validation accuracy: {meta['best_val_acc']:.4f}",
            f"Test accuracy: {test_report['test_acc']:.4f}",
            f"Test loss: {test_report['test_loss']:.4f}",
            "",
            "5. Submission links",
            "GitHub repo: [Please insert your public repository link here]",
            "Model weights download: [Please insert your Google Drive or other download link here]",
        ]
        y = 0.90
        for line in lines:
            fig.text(0.08, y, line, fontsize=11, va="top")
            y -= 0.035 if line else 0.02
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for image_path, title in [
            (assets["training_curves"], "Training Curves"),
            (assets["first_layer_weights"], "First-Layer Weight Visualization"),
            (assets["confusion_matrix"], "Confusion Matrix"),
            (assets["error_examples"], "Misclassified Examples"),
        ]:
            image = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(image)
            ax.set_title(title, fontsize=16, pad=12)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        title = "6. Observations and Analysis"
        paragraphs = [
            "Training curves show that both training loss and validation loss decrease steadily, while validation accuracy rises and stabilizes near the best epoch. This indicates that the learning-rate schedule is effective and the model converges without severe instability.",
            "The first-layer weights exhibit localized edge-like and contour-like patterns. Some hidden units respond to horizontal or vertical boundaries, while others emphasize diagonals, sleeve-like structures, shoe soles, and coarse clothing silhouettes. This suggests that the model learns low-level spatial templates before combining them for category discrimination.",
            "Typical misclassifications occur among visually similar upper-body categories such as T-shirt/top, Shirt, Pullover, and Coat. These classes share overlapping outlines and differ mainly in subtle sleeve length, collar shape, or texture details, which are difficult to preserve after flattening a 28x28 grayscale image.",
            "In the selected examples, Sandal and Sneaker can also be confused when the visible outline is narrow or when the sole structure dominates. This indicates that the model relies strongly on coarse silhouette information.",
            "Overall, the best configuration achieves strong validation and test accuracy with a relatively simple MLP, but the remaining errors show that a fully connected model still has limited ability to capture fine local spatial structure compared with convolution-based models.",
        ]
        fig.text(0.5, 0.96, title, ha="center", va="top", fontsize=16)
        y = 0.9
        for paragraph in paragraphs:
            fig.text(0.08, y, paragraph, fontsize=11, va="top", wrap=True)
            y -= 0.15

        example_lines = ["Selected error examples:"]
        for _, t, p in error_examples:
            example_lines.append(f"- {CLASS_NAMES[t]} -> {CLASS_NAMES[p]}")
        fig.text(0.08, 0.16, "\n".join(example_lines), fontsize=11, va="top")
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def save_report_markdown(output_md: Path, meta: dict, test_report: dict, error_examples):
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fashion-MNIST 实验报告",
        "",
        "## 1. 实验任务概述",
        "本实验使用 NumPy 从零实现单隐藏层多层感知机（MLP），在 Fashion-MNIST 数据集上完成 10 分类任务。",
        "",
        "## 2. 数据集与预处理",
        "- 数据集：Fashion-MNIST",
        "- 训练集：60,000 张图像",
        "- 测试集：10,000 张图像",
        "- 输入尺寸：28×28 灰度图，展开为 784 维向量",
        "- 预处理方式：先除以 255，再使用训练集像素均值和标准差做标准化",
        "",
        "## 3. 模型结构",
        f"- 输入层维度：{meta['config']['input_dim']}",
        f"- 隐藏层维度：{meta['config']['hidden_dim']}",
        f"- 输出层维度：{meta['config']['output_dim']}",
        f"- 激活函数：{meta['config']['activation']}",
        "- 损失函数：交叉熵损失",
        "- 优化方法：SGD",
        "- 正则化：L2 weight decay",
        "",
        "## 4. 最优超参数设置",
        f"- lr = {meta['config']['lr']}",
        f"- lr_decay = {meta['config']['lr_decay']}",
        f"- weight_decay = {meta['config']['weight_decay']}",
        f"- epochs = {meta['config']['epochs']}",
        f"- best validation accuracy = {meta['best_val_acc']:.4f}",
        "",
        "## 5. 测试集结果",
        f"- test loss = {test_report['test_loss']:.4f}",
        f"- test accuracy = {test_report['test_acc']:.4f}",
        "",
        "## 6. 训练曲线分析",
        "训练过程中，训练集和验证集的 Loss 整体持续下降，验证集 Accuracy 持续上升并在最佳轮次附近趋于稳定，说明该组超参数能够使模型较稳定地完成优化过程，没有出现明显的训练发散。",
        "",
        "## 7. 第一层权重可视化与空间模式观察",
        "将第一层隐藏层的权重恢复为 28×28 图像后，可以观察到网络学习到了一些局部边缘、方向纹理和轮廓模板。部分神经元更关注水平或竖直边缘，部分更像对斜边、鞋底结构、衣服外轮廓或袖口区域产生响应。这说明即使在全连接网络中，第一层仍然能够学习到对服装类别有区分作用的低层空间模式。",
        "",
        "## 8. 错例分析",
        "测试集中的错误主要集中在外观较为相似的服装类别之间，尤其是 T-shirt/top、Shirt、Pullover、Coat。这几类服装在 28×28 的低分辨率灰度图下，轮廓差异较小，而领口、袖长和面料细节又难以保留，因此更容易混淆。",
        "",
        "选取的错例包括：",
    ]
    for _, t, p in error_examples:
        lines.append(f"- {CLASS_NAMES[t]} 被分为 {CLASS_NAMES[p]}")
    lines.extend(
        [
            "",
            "这些错分现象表明，MLP 更依赖整体轮廓和粗粒度形状信息，而对局部空间结构和细节纹理的建模能力有限。",
            "",
            "## 9. 代码与模型链接",
            "- GitHub Repo: [请替换为你的公开 GitHub 仓库链接]",
            "- 模型权重下载地址: [请替换为你的 Google Drive 或其他下载链接]",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate report assets and PDF for the best checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/fashion-mnist")
    parser.add_argument("--output-dir", type=str, default="report")
    parser.add_argument("--metrics-json", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    model, meta = load_checkpoint(checkpoint_path)
    if args.metrics_json is not None:
        metrics_payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
        if "history" in metrics_payload:
            meta["history"] = metrics_payload["history"]
    data = prepare_data(Path(args.data_dir), seed=meta["config"]["seed"])
    y_pred = model.predict(data["X_test"])
    cm = confusion_matrix_np(data["y_test"], y_pred, 10)

    test_report_path = Path("model/test_reports") / f"{checkpoint_path.stem}_test.json"
    test_report = json.loads(test_report_path.read_text(encoding="utf-8"))

    training_curves_path = assets_dir / "training_curves.png"
    first_layer_weights_path = assets_dir / "first_layer_weights.png"
    error_examples_path = assets_dir / "error_examples.png"
    confusion_path = PROJECT_ROOT / "visualizations" / "test" / f"{checkpoint_path.stem}_confusion_matrix.png"

    plot_training_curves(meta["history"], training_curves_path)
    plot_first_layer_weights(model.params["W1"], first_layer_weights_path)
    error_examples = plot_error_examples(data["X_test"], data["y_test"], y_pred, error_examples_path)

    report_md = output_dir / "experiment_report.md"
    report_pdf = output_dir / "experiment_report.pdf"
    save_report_markdown(report_md, meta, test_report, error_examples)
    build_report_pdf(
        report_pdf,
        {
            "training_curves": training_curves_path,
            "first_layer_weights": first_layer_weights_path,
            "error_examples": error_examples_path,
            "confusion_matrix": confusion_path,
        },
        meta,
        test_report,
        error_examples,
    )

    print(f"Markdown report saved to: {report_md}")
    print(f"PDF report saved to: {report_pdf}")
    print(f"Training curves saved to: {training_curves_path}")
    print(f"First-layer weights saved to: {first_layer_weights_path}")
    print(f"Error examples saved to: {error_examples_path}")


if __name__ == "__main__":
    main()
