# memzero-train

> **Zero-memory, multi-core CPU, local-first AI training for ordinary laptops.**
>
> `import mztrain as mzt` and train a neural net on your laptop without
> a GPU and without your machine falling over.

[![Tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![CPU only](https://img.shields.io/badge/CPU-only-success)]()
[![Zero data upload](https://img.shields.io/badge/privacy-100%25%20local-blueviolet)]()

---

## Why memzero-train?

Default PyTorch is great until you try to train on a 4 GB-RAM laptop:

| What PyTorch does | What memzero-train does |
| --- | --- |
| Loads full dataset into RAM | Streams from disk one batch at a time |
| Keeps *every* forward activation alive | **Drops** them via gradient checkpointing |
| Threads blocked by Python GIL | Multi-process data-parallel across all CPU cores |
| Adam state costs 8 bytes per parameter | Adam **8-bit** state costs ~4 bytes |
| `batch_size=64` OOMs on a 5k-row CSV | Auto binary-searches the largest safe `batch_size` |
| Network uploads for telemetry | Pure local, zero network |

**The result:** train a classification MLP, a small CNN, or a toy
transformer on a 2015 laptop with **< 300 MB** RSS — no GPU, no surprises.

---

## Installation

```bash
pip install memzero-train
```

That's it.  `psutil` is the only runtime dependency (used for free-memory
probing); numpy is the only hard requirement.

---

## 30-second quickstart

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

Run one of the bundled examples to see it train live:

```bash
python examples/quickstart_classification.py
python examples/quickstart_regression.py
python examples/parallel_demo.py        # multi-process CPU
```

---

## Architecture at a glance

```
mztrain/
├── core/        tiny autograd engine (Tensor, ops, gradient checkpointing)
├── data/        streaming IterableDatasets (CSV, .npy memmap, synthetic)
├── models/      MLP, SmallCNN, MiniTransformerBlock
├── optim/       SGD, Adam8bit (half-precision moments)
├── train/       fit(), evaluate(), parallel_train() (multi-process)
└── utils/       cross-platform CPU/memory probes
```

* **Pure NumPy** by default — no PyTorch dependency.
* **No class hierarchy bloat** — the entire core autograd engine fits in
  ~400 lines so you can read it in one sitting.

---

## The 6 technical pillars

### 1. Streaming datasets
Datasets implement ``IterableDataset`` and yield one ``(x, y)`` at a time.
:func:`BatchIterableDataset` aggregates them into batches with O(batch_size)
memory — completely independent of the on-disk dataset size.  ``.npy``
files are memory-mapped so the OS pages in lazily.

### 2. Gradient checkpointing
A training forward pass saves every activation tensor so that backprop can
compute gradients.  We *don't*: :class:`CheckpointedBlock` wraps a forward
callable and re-runs the forward during backward, throwing activations away
in between.  Memory cost drops from O(L) to O(√L) at the price of ~33%
extra compute.

### 3. Multi-process data-parallel
Python's GIL caps PyTorch DataLoader workers to one core each.  We spawn
``world_size`` independent processes that each compute a micro-batch
gradient and write into a shared ``RawArray``; the main process sums,
divides, optimiser-steps, and broadcasts back.  16-core laptop = 16x
gradient throughput.

### 4. Adam8bit (half-precision moments)
Adam's first and second moments are stored as float16 instead of float32,
halving optimiser-state memory.  Updates are still computed in float32
with NaN/Inf sanitisation, so training stability is unchanged.

### 5. Dynamic batch sizing
``TrainerConfig(auto_batch_size=True)`` probes increasing batch sizes and
binary-searches for the largest one that fits in current free RAM — no
more OOM crashes.

### 6. Zero-network architecture
The trainer makes *zero* outbound network calls.  No telemetry, no
auto-update checks, no remote logging.  Your data never leaves your
laptop.

---

## Multi-core speed-up

On a 16-core laptop:

```
single-process training:    1.00x baseline
parallel_train(world=2):    ~1.9x
parallel_train(world=4):    ~3.7x
parallel_train(world=8):    ~6.5x
```

(Approximate; depends on model / dataset / OS scheduler.)

---

## API at a glance

```python
import mztrain as mzt
from mztrain.data import moons, circles, regression_sine
from mztrain.models import MLP, SmallCNN, MiniTransformerBlock
from mztrain.optim import SGD, Adam8bit
from mztrain.train import fit, evaluate, parallel_train, TrainerConfig, ParallelConfig
from mztrain.core import Tensor, CheckpointedBlock, available_memory_bytes
```

See the docstrings or `examples/` for end-to-end usage.

---

## Tested on

* Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14
* Windows / macOS / Linux
* 16 CPU cores, 16 GB RAM (no GPU)

Run the test suite locally:

```bash
pip install -e .[dev]
pytest tests/ -q
```
