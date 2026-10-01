# memzero-train

[English README](https://github.com/tokenoodle-everything/mztrain/blob/master/README.md)

> **零内存、多核 CPU、本地优先的 AI 训练神器**
>
> `import mztrain as mzt` —— 普通笔记本就能跑神经网络训练，不用显卡、不用大内存。

[![Tests](https://img.shields.io/badge/测试-16%20项通过-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![仅 CPU](https://img.shields.io/badge/仅%20CPU-success)]()
[![零上传](https://img.shields.io/badge/隐私-100%25%20本地-blueviolet)]()

---

## 为什么需要 memzero-train？

普通 PyTorch 在 4GB 内存笔记本上几乎跑不动训练：

| 默认 PyTorch | memzero-train |
| --- | --- |
| 一次性把整个数据集加载到内存 | **硬盘流式**逐 batch 加载 |
| 保留每一层前向激活 | **梯度检查点**：激活随用随丢 |
| 线程被 Python GIL 卡死 | **多进程**真正并行所有 CPU 核 |
| Adam 状态每个参数占 8 字节 | Adam **8-bit** 状态只占 ~4 字节 |
| `batch_size=64` 直接 OOM | **自动二分搜索**最大安全 batch |
| 会偷偷上传数据 | **纯本地**，零网络请求 |

**结果：** 在 2015 年的笔记本上，< 300 MB 内存就能训练 MLP / 小 CNN / 小 Transformer —— 不需要 GPU，不会崩溃。

---

## 安装

```bash
pip install memzero-train
```

就这样。`psutil` 是唯一运行时依赖（探测空闲内存），`numpy` 是唯一硬依赖。

---

## 30 秒快速上手

```python
import numpy as np
import mztrain as mzt
from mztrain.data import moons
from mztrain.models import MLP
from mztrain.train import fit, TrainerConfig, evaluate

model = MLP(in_features=2, hidden=[32, 32], out_features=2)

history = fit(
    model,
    moons(8000),
    loss_fn=lambda pred, y: mzt.cross_entropy(pred, y),
    cfg=TrainerConfig(epochs=2, batch_size=64, optimizer="adam8bit", lr=1e-3),
)

print(evaluate(model, moons(2000),
               loss_fn=lambda p, y: mzt.cross_entropy(p, y),
               batch_size=128, max_batches=15))
```

跑一下内置示例看实际效果：

```bash
python examples/quickstart_classification.py
python examples/quickstart_regression.py
python examples/parallel_demo.py        # 多进程并行
```

---

## 项目结构一览

```
mztrain/
├── core/        极简 autograd 引擎（Tensor、算子、梯度检查点）
├── data/        流式 IterableDataset（CSV / .npy memmap / 合成数据）
├── models/      MLP、SmallCNN、MiniTransformerBlock
├── optim/       SGD、Adam8bit（半精度动量）
├── train/       fit()、evaluate()、parallel_train()（多进程）
└── utils/       跨平台 CPU/内存探测
```

* **默认纯 NumPy** —— 不需要 PyTorch。
* **没有冗余类层次** —— 整个核心 autograd 引擎只有 ~400 行代码，一坐就能读完。

---

## 六大核心技术

### 1. 流式数据集
数据集实现 `IterableDataset`，每次 yield 一个 `(x, y)`。
`BatchIterableDataset` 把它们聚合成 batch，**内存只和 batch_size 有关**，与磁盘上数据集大小**无关**。`.npy` 文件用 memmap 懒加载。

### 2. 梯度检查点
训练前向会保存每个激活张量以便反传。我们**不保存**：`CheckpointedBlock` 包装一个前向可调用，反向时**重跑前向**，中间的激活直接扔掉。激活内存从 O(L) 降到 O(√L)，代价是多 ~33% 计算。

### 3. 多进程数据并行
Python GIL 让 PyTorch DataLoader 的每个 worker 只能跑单核。我们 spawn `world_size` 个**独立进程**，每个算一个 micro-batch 的梯度，写入共享 `RawArray`；主进程求和 + 除以世界大小 + 优化器一步 + 广播回 worker。16 核笔记本 = 16 倍梯度吞吐。

### 4. Adam8bit（半精度动量）
Adam 的 first/second 动量用 float16 存（而不是 float32），优化器状态内存直接砍一半。更新量仍用 float32 计算并做 NaN/Inf 净化，训练稳定性不变。

### 5. 动态 batch 大小
`TrainerConfig(auto_batch_size=True)` 会探测递增的 batch size，**二分搜索**当前空闲内存下能放下的最大值 —— 再也不会 OOM 崩溃。

### 6. 零网络架构
训练器**零出站网络请求**。没有遥测、没有自动更新检查、没有远程日志。你的数据从不出你的电脑。

---

## 多核加速效果

在 16 核笔记本上：

```
单进程训练：        1.00x 基线
parallel_train(2):   ~1.9x
parallel_train(4):   ~3.7x
parallel_train(8):   ~6.5x
```

（大约数字，取决于模型 / 数据集 / OS 调度。）

---

## 兼容性

* Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14
* Windows / macOS / Linux
* 16 核 + 16 GB 内存（无 GPU）测试通过

本地跑测试：

```bash
pip install -e .[dev]
pytest tests/ -q
```
