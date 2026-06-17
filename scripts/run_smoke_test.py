#!/usr/bin/env python
"""
Quick CLI smoke test for the A3 attention implementation.

Usage:
    python scripts/run_smoke_test.py

Builds a small Transformer encoder under each of the four attention modes
(softmax, temp_softmax, sparsemax, a3) and verifies:
  - the forward pass runs without error
  - output shapes are correct
  - attention weights are valid probability distributions in every layer
  - reports the parameter overhead of each mode relative to softmax
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from a3_attention import (
    A3TransformerEncoder,
    ATTN_MODES,
    set_seed,
    count_parameters,
)


def main():
    print("=" * 60)
    print("A3 Attention -- Smoke Test")
    print("=" * 60)

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B, n, d_model = 4, 32, 128
    vocab_size = 1000
    num_classes = 3
    num_heads = 4
    num_layers = 3

    dummy_ids = torch.randint(1, vocab_size, (B, n)).to(device)

    param_counts = {}
    for mode in ATTN_MODES:
        model = A3TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=256,
            max_seq_len=n,
            num_classes=num_classes,
            mode=mode,
            d_e=16,
            dropout=0.0,
        ).to(device)

        logits, cls_repr, attn_weights = model(dummy_ids)

        assert logits.shape == (B, num_classes), f"[{mode}] bad logits shape"

        for ell, attn in enumerate(attn_weights):
            assert attn.shape == (B, n, n), f"[{mode}] layer {ell} bad attn shape"
            assert (attn >= 0).all(), f"[{mode}] layer {ell}: negative attention!"
            row_sums = attn.sum(dim=-1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-4
            ), f"[{mode}] layer {ell}: rows do not sum to 1!"

        n_params = count_parameters(model)
        param_counts[mode] = n_params
        print(
            f"  mode={mode:13s} | params={n_params:>8,} | "
            f"logits shape={tuple(logits.shape)} | all checks passed"
        )

    print("\nParameter overhead relative to softmax:")
    base = param_counts["softmax"]
    for mode, n_params in param_counts.items():
        delta = n_params - base
        pct = 100 * delta / base if base else 0.0
        print(f"  {mode:13s}: {n_params:>8,}  (delta={delta:+,}, {pct:+.2f}%)")

    print("\nAll smoke tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
