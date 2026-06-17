"""
A3 Attention: Adaptive Activation Attention.

A learnable, input-conditioned replacement for the fixed softmax in
Transformer attention. See paper/a3_paper.pdf for the full empirical study.
"""

from .a3_attention import (
    Sparsemax,
    SoftmaxAttention,
    TempSoftmaxAttention,
    SparsemaxAttention,
    A3Attention,
    MultiHeadAttention,
    A3TransformerBlock,
    A3TransformerEncoder,
    ATTN_MODES,
    set_seed,
    count_parameters,
    build_optimizer,
    get_lr_scheduler,
    train_one_epoch,
    evaluate,
    analyze_gates,
)

__version__ = "1.0.0"
__all__ = [
    "Sparsemax",
    "SoftmaxAttention",
    "TempSoftmaxAttention",
    "SparsemaxAttention",
    "A3Attention",
    "MultiHeadAttention",
    "A3TransformerBlock",
    "A3TransformerEncoder",
    "ATTN_MODES",
    "set_seed",
    "count_parameters",
    "build_optimizer",
    "get_lr_scheduler",
    "train_one_epoch",
    "evaluate",
    "analyze_gates",
]
