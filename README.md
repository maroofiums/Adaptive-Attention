# Adaptive Activation Attention ($A^3$)

[![Build Status](https://img.shields.io/badge/build-passing-success?style=flat-square)](https://github.com/maroofiums/A3-Attention)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue?style=flat-square)](pyproject.toml)
[![PyTorch Version](https://img.shields.io/badge/pytorch-%E2%89%A52.0.0-orange?style=flat-square)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/status-under%20review-yellow?style=flat-square)](CITATION.cff)

This repository contains the official PyTorch implementation and empirical evaluation of **Adaptive Activation Attention ($A^3$)**, an architectural framework that replaces the fixed softmax normalization in Transformer attention layers with a learnable, input-conditioned activation network.

---

### Core Research Metadata
* **Paper Title:** [Is Learnable Attention Normalization Beneficial? An Empirical Investigation with Mixed Findings](https://zenodo.org/records/20364414)
* **Author:** Muhammad Maroof Farooq 
* **Date:** April 2026
* **Manuscript Document:** `paper/a3_paper.pdf`
* **Status:** Workshop submission, under review

---

## 1. Architectural Concept

The canonical scaled dot-product attention mechanism relies on a fixed softmax operation applied uniformly across every head, layer, and sequential token. This design imposes a strict, static mathematical layout over token similarity scores, which can restrict a model's capacity to adapt its operational attention profile to diverse structural configurations or complex linguistic contexts.

**Adaptive Activation Attention ($A^3$)** addresses this by modeling attention normalization via a lightweight, gated sub-network. This network accepts the raw score distribution, positional identifiers, and layer-specific state vectors as inputs, dynamically mapping similarity values to valid attention weights that strictly satisfy non-negativity and row-stochastic (summation-to-one) boundaries.

A comprehensive diagram outlining the internal tensor operations and layer-wise interactions is located at `assets/architecture_diagram.png`.

### Methodological Rigor and Open Science
In alignment with the principles of reproducible machine learning research, this project explicitly presents **mixed findings**. While $A^3$ uncovers targeted representational benefits in specialized task environments-such as long-sequence context windows-it does not consistently outperform a carefully tuned, parameter-matched baseline configuration across all standard natural language benchmarks (e.g., typical BERT-scale token environments). Full analytical details are presented in the [Empirical Benchmarks](#4-empirical-benchmarks) section below.

---

## 2. Supported Attention Modes

The core library located inside `src/a3_attention.py` includes **four operational attention variants** used for comparative structural evaluation:

* **`softmax`**
  Standard PyTorch Scaled Dot-Product Attention functioning as the architectural control baseline.
* **`a3_gated`**
  The baseline $A^3$ framework, which feeds the underlying score matrices through an input-conditioned gated transformation to generate contextual weights.
* **`a3_residual`**
  An experimental mode applying a learnable, context-dependent adjustment mapping as a structured residual bypass over a traditional softmax framework.
* **`a3_learned_baseline`**
  A parameter-matched control mode where layer and head normalization values are static throughout evaluation. They are updated globally via backpropagation but do not condition dynamically on specific token sequences at runtime.

---

## 3. Installation & Usage

### Setup Environment
Clone the repository workspace and install all runtime and development requirements using `requirements.txt`:

```bash
git clone https://github.com/maroofiums/A3-Attention.git
cd A3-Attention
pip install -r requirements.txt

```

Alternatively, you can install the local workspace package in editable distribution mode via `pyproject.toml`:

```bash
pip install -e .

```

### Programmatic Integration

```python
import torch
from a3_attention import AdaptiveActivationAttention

# Initialize model dimensions
batch_size = 2
num_heads = 4
seq_len = 8
embed_dim = 64

# Instantiate the A3 Attention layer block
a3_layer = AdaptiveActivationAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    attention_mode="a3_gated",  # Selectable: 'softmax', 'a3_gated', 'a3_residual', 'a3_learned_baseline'
    layer_idx=0
)

# Simulated input sequence tensor (Batch Size, Sequence Length, Embedding Dimension)
x = torch.randn(batch_size, seq_len, embed_dim)

# Execute forward processing pass
output, attention_weights = a3_layer(x)

print("Output matrix shape:", output.shape)              # Expected: torch.Size([2, 8, 64])
print("Attention weights shape:", attention_weights.shape)  # Expected: torch.Size([2, 4, 8, 8])

```

---

## 4. Empirical Benchmarks

Performance testing incorporates paired bootstrap resampling utilizing 10,000 unique iterations to accurately quantify statistical variance and cross-run trends.

| Attention Mechanism | GLUE Benchmark (Avg Score) | SCROLLS (Long Context) | WMT-14 En-De (BLEU) | Parameter Overhead |
| --- | --- | --- | --- | --- |
| **Standard Softmax** | **81.4** | 42.1 | **27.3** | *0 (Baseline)* |
| **$A^3$-Gated** | 81.2 | **43.8** | 26.9 | +1.2% |
| **$A^3$-Residual** | 81.5 | 42.9 | 27.1 | +0.8% |
| **Learned Baseline** | 80.9 | 41.8 | 26.5 | +0.2% |

### Core Analysis

* **Long-Context Behavior:** The `a3_gated` paradigm exhibits distinct, statistically relevant improvements across extended sequence configurations (**SCROLLS**). This indicates that dynamic activation constraints assist in preventing attention dispersion or dilution when processing dense sequence structures.
* **Standard Downstream Evaluations:** Across standard localized natural language structures like **GLUE** and **WMT-14**, the additional parameter density introduced by the $A^3$ sub-network falls within standard control deviations, demonstrating that traditional softmax remains a highly optimal runtime option for standard sequence bounds.

---

## 5. Verification Suite

Code validation is managed through an integrated testing setup designed to confirm specific architectural invariants. The system contains **40 automated unit tests** tracking output matrix shapes, non-negativity parameters, row-stochastic bounds, and stable, non-NaN gradient execution paths.

Execute the thorough test suite via `pytest`:

```bash
pytest tests/ -v

```

Execute a fast, minimal structural check via the provided smoke script:

```bash
python scripts/run_smoke_test.py

```

To review an interactive, step-by-step mathematical walkthrough detailing tensor workflows and underlying derivations, execute the project Jupyter notebook:

```bash
jupyter notebook notebooks/A3_Adaptive_Attention.ipynb

```

---

## 6. Contribution Guidelines

Contributions focusing on scaling validations up to large model topologies or highlighting architectural edge cases are welcome.

Please read `CONTRIBUTING.md` before initiating a Pull Request. Current development roadmap items include:

* **Pre-Training Scaling Pass:** Evaluating $A^3$ structures within foundational BERT-base or GPT-2 model pipelines to track impacts on global pre-training objectives.
* **Symmetric Parameter Constraints:** Introducing alternative baseline control setups where parameters matching $A^3$ are allocated directly inside standard feed-forward blocks to thoroughly isolate systemic structural variance.
* **Alternative Mathematical Baselines:** Extending comparative profiling by incorporating $\alpha$-entmax structures using the `entmax` module interface.

---

## 7. License

This project is licensed under the conditions of the **MIT License**. For granular specifications, view the terms in `LICENSE`.

---

