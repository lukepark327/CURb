# CURb

**Hybrid-Domain Anchored CUR Decomposition for Rehearsal-Free Continual Fine-Tuning**

## Overview

CURb is a parameter-efficient fine-tuning (PEFT) framework for **rehearsal-free continual learning**.
It leverages CUR matrix decomposition to select structurally important columns (**C**) and rows (**R**)
from pretrained weight matrices as frozen anchors, and trains only the compact **U** linkage matrix
(r x r) as a lightweight adapter.

### Key Features

- **CUR-based adapter** with three basis construction modes: `weight`, `cov_fast` (activation-aware), and `hybrid`
- **DEIM-based importance-aware** row/column selection for structured basis anchoring
- **Spectral-norm alpha scaling** and optional whitening for stable training across layers
- **Baseline implementations** for fair comparison: LoRA, MoRA, CURLoRA, oLoRA, InfLoRA, BiLoRA, LoRAC

### Supported Benchmarks

| Benchmark | Tasks | Runner |
|---|---|---|
| LLM Continual Learning | BoolQ, WinoGrande, ARC-E/C, PIQA, OBQA, SocialIQA, LogiQA | `experiments/llm/cl.py` |
| LLM Text Classification CL | DBpedia, Amazon, Yahoo, AG News | `experiments/llm/tc_cl.py` |
| LLM UUID Memorization | UUID-based continual learning | `experiments/llm/uuid_cl.py` |
| ViT Class-Incremental | CIFAR-100, ImageNet subsets | `experiments/vit/vit_cl.py` |
| ViT Domain-Incremental | DomainNet (6 domains) | `experiments/vit/vit_dl.py` |

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0 with CUDA support
- NVIDIA GPU with >= 24 GB VRAM (for LLM experiments)

### Setup

```bash
# Clone the repository
git clone https://github.com/<anonymous>/CURb.git
cd CURb

# Install dependencies
pip install -r requirements.txt
```

#### Optional: MoRA baseline

To run MoRA comparisons, clone and install the MoRA fork of PEFT:

```bash
git clone https://github.com/kongds/MoRA.git
cd MoRA/peft-mora && pip install -e . && cd ../..
```

## Quick Start

### Using CURb as an adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from curb import inject_curb, merge_curb, strip_curb, freeze_except_curb_U
from curb_basis import load_or_build_curb_basis

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Build CUR basis from calibration data
basis = load_or_build_curb_basis(
    model=model,
    tokenizer=tokenizer,
    mode="weight",             # "weight" | "cov_fast" | "hybrid"
    rank=256,
    deim_importance_order="low",
)

# Inject CURb adapters
inject_curb(
    model, basis,
    layer_indices=list(range(32)),
    ffn_module_names=["gate_proj"],
    attn_module_names=["q_proj", "k_proj"],
    alpha=1.0,
)

# Freeze all except U matrices and train
freeze_except_curb_U(model)
# ... training loop ...

# Merge adapters back into base weights
merge_curb(model)

# Strip CURb modules (revert to plain nn.Linear)
strip_curb(model)
```

### Running LLM continual learning experiments

```bash
# From the repository root:
bash scripts/run_llm_cl.sh
```

See `scripts/` for all available experiment launchers.

## Repository Structure

```
CURb/
├── methods/
│   ├── curb/
│   │   ├── curb.py            # CURbLinear module, inject/merge/strip utilities
│   │   └── curb_basis.py      # CUR basis construction and whitening
│   └── baselines/
│       ├── curlora.py         # CUR-based LoRA variant
│       ├── bilora.py          # Fourier-domain sparse LoRA (BiLoRA)
│       ├── olora.py           # Orthogonal LoRA for CL (oLoRA)
│       ├── inflora.py         # DualGPM-based CL (InfLoRA)
│       └── lorac.py           # Composition-based CL (LoRAC)
├── cur_utils/                 # CUR decomposition primitives
│   ├── cur.py                 # DEIM-based column/row selection
│   └── cur_models.py          # Activation hooking utilities
├── experiments/
│   ├── llm/                   # LLM experiment runners
│   │   ├── cl.py              # Multi-task CL benchmark
│   │   ├── tc_cl.py           # Text classification CL
│   │   └── uuid_cl.py         # UUID memorization CL
│   └── vit/                   # Vision Transformer experiments
│       ├── vit_cl.py          # Class-incremental learning
│       └── vit_dl.py          # Domain-incremental learning
├── scripts/                   # Experiment launcher scripts
│   ├── run_llm_cl.sh
│   ├── run_vit_cl.sh
│   ├── run_vit_dl.sh
│   ├── diag_curb_llm.sh
│   ├── diag_curb_vit.sh
│   └── plot_uuid_compare.py
├── curb.py                    # Convenience re-export
├── curb_basis.py              # Convenience re-export
├── requirements.txt
└── README.md
```

## CURb Method Details

CURb decomposes the adapter update as:

```
output = W @ x + alpha * (C @ U @ R) @ x
```

where:
- **W** is the frozen pretrained weight matrix
- **C** (m x r) contains selected columns of W
- **R** (r x n) contains selected rows of W
- **U** (r x r) is the only trainable parameter
- **alpha** scales the adapter contribution

Basis modes:
- `weight`: Selects C and R based on weight magnitude via DEIM
- `cov_fast`: Uses activation covariance statistics for importance-aware selection
- `hybrid`: Combines weight and activation information

## Citation

> Paper under review.
