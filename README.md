# Fashion-MNIST HW1

## 目录结构

```text
cv-hw1/
├─ data/
│  └─ fashion-mnist/              # 数据集
├─ model/
│  ├─ weights/                    # 模型权重
│  ├─ train_metrics/              # 训练历史
│  ├─ search_results/             # 超参数搜索结果
│  └─ test_reports/               # 测试结果
├─ report/
│  ├─ assets/                     # 报告用图片
│  ├─ experiment_report.md        # 实验报告
│  └─ experiment_report.pdf       # 实验报告 PDF
├─ scripts/
│  ├─ train.py                    # 训练脚本
│  ├─ search_params.py            # 超参数搜索脚本
│  ├─ test.py                     # 测试脚本
│  └─ generate_report.py          # 报告生成脚本
├─ src/
│  └─ models/
│     └─ mlp.py                   # MLP 模型
├─ visualizations/
│  ├─ train/                      # 训练可视化
│  └─ test/                       # 测试可视化
├─ requirements.txt
└─ HW1_计算机视觉 (1).pdf
```

## 操作说明

安装依赖：

```powershell
pip install -r requirements.txt
```

训练模型：

```powershell
python scripts/train.py --run-name baseline --hidden-dim 512 --activation relu --epochs 50 --lr 0.2 --lr-decay 0.95 --weight-decay 0.001
```

超参数搜索：

```powershell
python scripts/search_params.py
```

测试最优模型：

```powershell
python scripts/test.py --checkpoint model/weights/hd512_relu_lr0p2_decay0p95_wd0p001_best.npz
```

生成实验报告：

```powershell
python scripts/generate_report.py --checkpoint model/weights/hd512_relu_lr0p2_decay0p95_wd0p001_best.npz --metrics-json model/train_metrics/report_best_full50.json
```
