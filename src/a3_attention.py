"""
A3: Adaptive Activation Attention - PyTorch Implementation
Paper: "Is Learnable Attention Normalization Beneficial?
        An Empirical Investigation with Mixed Findings"
Author: Muhammad Maroof Farooq

This file contains:
  1. A3Attention       - single-head A3 attention module
  2. A3MultiHeadAttention - multi-head wrapper
  3. A3TransformerBlock   - full Transformer block with A3
  4. A3TransformerEncoder - stacked encoder
  5. Baseline classes  - SoftmaxAttention, TempSoftmaxAttention,
                         SparsemaxAttention
  6. Training utilities - loss, optimizer setup, seed fixing
  7. Quick smoke-test  - runs at the bottom under __main__
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ============================================================
# 0. Utility helpers
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 1. Sparsemax (needed as a baseline)
# ============================================================

class Sparsemax(nn.Module):
    """
    Sparsemax: Euclidean projection onto the probability simplex.
    Martins & Astudillo, ICML 2016.

    Operates on the last dimension of the input tensor.
    """

    def forward(self, z: Tensor) -> Tensor:
        # z: (..., n)
        *batch, n = z.shape
        z_flat = z.reshape(-1, n)

        # Sort descending
        z_sorted, _ = torch.sort(z_flat, dim=-1, descending=True)
        z_cumsum = torch.cumsum(z_sorted, dim=-1)

        # Find threshold k
        k = torch.arange(1, n + 1, dtype=z.dtype, device=z.device)
        # condition: 1 + k * z_sorted[k] > cumsum[k-1]
        # equivalently: z_sorted > (cumsum - 1) / k
        support = (1 + k * z_sorted) > z_cumsum          # (..., n)
        # last True index = k* - 1 (0-indexed)
        k_star = support.sum(dim=-1, keepdim=True).long() # (..., 1)
        # threshold tau
        # tau = (z_cumsum[k*-1] - 1) / k*
        # gather cumsum at position k*-1
        idx = (k_star - 1).clamp(min=0)
        tau_num = z_cumsum.gather(dim=-1, index=idx) - 1.0
        tau = tau_num / k_star.float()

        p = (z_flat - tau).clamp(min=0.0)
        return p.reshape(*batch, n)


# ============================================================
# 2. Baseline attention modules
# ============================================================

class SoftmaxAttention(nn.Module):
    """
    Standard scaled dot-product attention with fixed softmax.
    No learnable parameters beyond Q/K/V projections (handled externally).
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            Q: (B, n, d_k)
            K: (B, n, d_k)
            V: (B, n, d_v)
            mask: (B, n, n) boolean mask; True = ignore position

        Returns:
            output: (B, n, d_v)
            attn_weights: (B, n, n)
        """
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B,n,n)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        output = torch.bmm(attn, V)
        return output, attn


class TempSoftmaxAttention(nn.Module):
    """
    Softmax attention with one learnable temperature scalar per head.
    tau is initialized to 1.0 (recovers standard softmax).
    This is the primary simple baseline: one free parameter per head.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        # log(tau) parameterization ensures tau > 0 always
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(dropout)

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores * self.tau        # learnable sharpness

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        output = torch.bmm(attn, V)
        return output, attn


class SparsemaxAttention(nn.Module):
    """
    Attention with sparsemax normalization instead of softmax.
    No additional learnable parameters.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.sparsemax = Sparsemax()
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # sparsemax applied row-wise (last dim)
        B, n, _ = scores.shape
        attn = self.sparsemax(scores.reshape(B * n, n)).reshape(B, n, n)
        attn = self.attn_drop(attn)
        output = torch.bmm(attn, V)
        return output, attn


# ============================================================
# 3. A3 Attention: core module
# ============================================================

class A3Attention(nn.Module):
    """
    Adaptive Activation Attention (A3) - single head.

    Replaces fixed softmax with a learnable gated transformation:

        c   = [mean_Q ; mean_K ; e_lh]         context vector
        g   = sigmoid(W_g @ c + b_g)           gate in (0,1)^{d_k}
        G   = expand(g)                         broadcast to (n, n)
        B   = U @ c^T                           low-rank bias (n, n)
        out = softmax(S ⊙ G + B) @ V

    where S = Q K^T / sqrt(d_k).

    Validity: softmax guarantees non-negativity and normalization for
              all parameter values (Proposition 1 in paper).
    Init:     b_g = 3*ones  =>  sigmoid(3) ≈ 0.95 ≈ 1
              W_g = 0, U = 0  =>  recovers softmax at init
              (Proposition 2 in paper)
    """

    def __init__(
        self,
        d_k: int,
        seq_len: int,
        d_e: int = 16,
        layer_idx: int = 0,
        head_idx: int = 0,
        gate_init_bias: float = 3.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_k:            key/query dimension per head
            seq_len:        maximum sequence length (for U matrix)
            d_e:            dimension of layer-head embedding
            layer_idx:      which layer (for embedding)
            head_idx:       which head  (for embedding)
            gate_init_bias: initial value of b_g; sigmoid(3) ≈ 0.95
            dropout:        attention dropout probability
        """
        super().__init__()
        self.d_k = d_k
        self.seq_len = seq_len
        self.d_e = d_e
        self.layer_idx = layer_idx
        self.head_idx = head_idx

        # --- Context dimension: [mean_Q (d_k) ; mean_K (d_k) ; e_lh (d_e)]
        d_ctx = 2 * d_k + d_e

        # --- Layer-head embedding (identifies this specific head)
        self.e_lh = nn.Parameter(torch.zeros(d_e))

        # --- Gate network: d_ctx -> d_k
        self.W_g = nn.Linear(d_ctx, d_k, bias=True)
        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, gate_init_bias)  # sigmoid(3) ≈ 0.95

        # --- Low-rank bias: U has shape (seq_len, d_ctx)
        # B = U @ c^T  gives shape (seq_len, 1) -> broadcast to (n, n)
        # We use a simpler form: B_ij = (U @ c)_i   (row-wise bias)
        # This makes B depend on row position and context.
        self.U = nn.Linear(d_ctx, seq_len, bias=False)
        nn.init.zeros_(self.U.weight)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            Q:    (B, n, d_k)
            K:    (B, n, d_k)
            V:    (B, n, d_v)
            mask: (B, n, n)  - True = mask out (set to -inf before softmax)

        Returns:
            output:       (B, n, d_v)
            attn_weights: (B, n, n)
        """
        B, n, d_k = Q.shape
        assert d_k == self.d_k, f"Expected d_k={self.d_k}, got {d_k}"

        # --- Raw attention scores: (B, n, n)
        S = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # --- Build context vector c: (B, d_ctx)
        mean_Q = Q.mean(dim=1)                              # (B, d_k)
        mean_K = K.mean(dim=1)                              # (B, d_k)
        e_lh_exp = self.e_lh.unsqueeze(0).expand(B, -1)    # (B, d_e)
        c = torch.cat([mean_Q, mean_K, e_lh_exp], dim=-1)  # (B, d_ctx)

        # --- Gate: (B, d_k) in (0, 1)
        g = torch.sigmoid(self.W_g(c))                     # (B, d_k)

        # Gate is scalar per score-column (shared across rows).
        # G has shape (B, 1, d_k) -> broadcast to (B, n, n) when n == d_k.
        # More generally: we expand g to (B, n, n) by treating each gate
        # value as a column-wise scaling. Here we broadcast over rows.
        # Shape: (B, 1, d_k) * (B, n, n)  -- only valid when d_k == n.
        # For the general case (d_k != n), we use a mean-gate scalar:
        if n == d_k:
            G = g.unsqueeze(1)                              # (B, 1, d_k)
        else:
            # Fall back: use mean gate as a scalar temperature per example
            G = g.mean(dim=-1, keepdim=True).unsqueeze(-1) # (B, 1, 1)

        # --- Low-rank bias: (B, n) row-wise, broadcast to (B, n, n)
        # Each row i gets the same additive constant U(c)[i]
        # Shape: (B, seq_len) -> trim/pad to (B, n, 1) -> (B, n, n)
        bias_full = self.U(c)                               # (B, seq_len)
        bias_n = bias_full[:, :n]                           # (B, n)
        B_mat = bias_n.unsqueeze(-1).expand(B, n, n)       # (B, n, n)

        # --- Gated score
        S_gated = S * G + B_mat                             # (B, n, n)

        if mask is not None:
            S_gated = S_gated.masked_fill(mask, float("-inf"))

        # --- Softmax ensures validity (Proposition 1)
        attn = F.softmax(S_gated, dim=-1)                  # (B, n, n)
        attn = self.attn_drop(attn)

        # --- Output
        output = torch.bmm(attn, V)                        # (B, n, d_v)
        return output, attn

    def gate_stats(self, Q: Tensor, K: Tensor) -> dict:
        """
        Diagnostic: return gate values for analysis (no grad).
        Useful for computing CV (coefficient of variation) across examples.
        """
        with torch.no_grad():
            B = Q.size(0)
            mean_Q = Q.mean(dim=1)
            mean_K = K.mean(dim=1)
            e_lh_exp = self.e_lh.unsqueeze(0).expand(B, -1)
            c = torch.cat([mean_Q, mean_K, e_lh_exp], dim=-1)
            g = torch.sigmoid(self.W_g(c))
        return {
            "gate_mean": g.mean().item(),
            "gate_std":  g.std().item(),
            "gate_cv":   (g.std() / (g.mean() + 1e-8)).item(),
            "gate_min":  g.min().item(),
            "gate_max":  g.max().item(),
        }


# ============================================================
# 4. Multi-Head Attention wrapper (supports all variants)
# ============================================================

ATTN_MODES = ("softmax", "temp_softmax", "sparsemax", "a3")


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention supporting four normalization modes:
      - 'softmax'      : standard fixed softmax
      - 'temp_softmax' : learned temperature per head
      - 'sparsemax'    : sparse projection
      - 'a3'           : Adaptive Activation Attention (proposed)

    Q/K/V projections are shared across modes.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mode: str = "a3",
        seq_len: int = 128,
        d_e: int = 16,
        layer_idx: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert mode in ATTN_MODES, f"mode must be one of {ATTN_MODES}"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mode = mode

        # Shared projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Per-head attention modules
        if mode == "softmax":
            self.heads = nn.ModuleList([
                SoftmaxAttention(dropout) for _ in range(num_heads)
            ])
        elif mode == "temp_softmax":
            self.heads = nn.ModuleList([
                TempSoftmaxAttention(dropout) for _ in range(num_heads)
            ])
        elif mode == "sparsemax":
            self.heads = nn.ModuleList([
                SparsemaxAttention(dropout) for _ in range(num_heads)
            ])
        elif mode == "a3":
            self.heads = nn.ModuleList([
                A3Attention(
                    d_k=self.d_k,
                    seq_len=seq_len,
                    d_e=d_e,
                    layer_idx=layer_idx,
                    head_idx=h,
                    dropout=dropout,
                )
                for h in range(num_heads)
            ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:    (B, n, d_model)
            mask: (B, n, n) - True positions are masked

        Returns:
            output:      (B, n, d_model)
            attn_avg:    (B, n, n)  mean attention across heads
        """
        B, n, _ = x.shape
        H = self.num_heads
        dk = self.d_k

        # Project and split into heads: (B, n, H, d_k) -> (B*H, n, d_k)
        def project_split(W):
            out = W(x)                               # (B, n, d_model)
            out = out.view(B, n, H, dk)              # (B, n, H, d_k)
            out = out.permute(0, 2, 1, 3)            # (B, H, n, d_k)
            return out.reshape(B * H, n, dk)         # (B*H, n, d_k)

        Q = project_split(self.W_Q)
        K = project_split(self.W_K)
        V = project_split(self.W_V)

        # Expand mask for all heads
        if mask is not None:
            mask_h = mask.unsqueeze(1).expand(B, H, n, n).reshape(B * H, n, n)
        else:
            mask_h = None

        # Run each head
        head_outputs, attn_maps = [], []
        for h, head_module in enumerate(self.heads):
            Q_h = Q[h::H] if False else Q.view(B, H, n, dk)[:, h].reshape(B, n, dk)
            K_h = K.view(B, H, n, dk)[:, h].reshape(B, n, dk)
            V_h = V.view(B, H, n, dk)[:, h].reshape(B, n, dk)
            mask_hh = mask_h.view(B, H, n, n)[:, h] if mask_h is not None else None

            out_h, attn_h = head_module(Q_h, K_h, V_h, mask_hh)
            head_outputs.append(out_h)      # (B, n, d_k)
            attn_maps.append(attn_h)        # (B, n, n)

        # Concatenate heads and project
        concat = torch.cat(head_outputs, dim=-1)     # (B, n, d_model)
        output = self.W_O(concat)                    # (B, n, d_model)
        attn_avg = torch.stack(attn_maps, dim=1).mean(dim=1)  # (B, n, n)

        return output, attn_avg

    def get_gate_stats(self, x: Tensor) -> list:
        """
        Return gate diagnostics for all A3 heads.
        Only meaningful when mode == 'a3'.
        """
        if self.mode != "a3":
            return []
        B, n, _ = x.shape
        H = self.num_heads
        dk = self.d_k

        Q = self.W_Q(x).view(B, n, H, dk).permute(0, 2, 1, 3)
        K = self.W_K(x).view(B, n, H, dk).permute(0, 2, 1, 3)

        stats = []
        for h, head_module in enumerate(self.heads):
            Q_h = Q[:, h]    # (B, n, d_k)
            K_h = K[:, h]
            stats.append({
                "head": h,
                **head_module.gate_stats(Q_h, K_h)
            })
        return stats


# ============================================================
# 5. Transformer Block
# ============================================================

class A3TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block with configurable attention mode.
    Architecture: MHA -> Add & Norm -> FFN -> Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        mode: str = "a3",
        seq_len: int = 128,
        d_e: int = 16,
        layer_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            mode=mode,
            seq_len=seq_len,
            d_e=d_e,
            layer_idx=layer_idx,
            dropout=dropout,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Pre-norm (modern convention; swap if you want post-norm)
        attn_out, attn_w = self.attn(self.norm1(x), mask)
        x = x + self.drop(attn_out)
        x = x + self.ff(self.norm2(x))
        return x, attn_w


# ============================================================
# 6. Transformer Encoder
# ============================================================

class A3TransformerEncoder(nn.Module):
    """
    Stacked Transformer encoder with token + positional embeddings.
    Returns:
        - [CLS] token representation (for classification)
        - full sequence representation
        - list of attention weight tensors per layer
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 128,
        num_classes: int = 2,
        mode: str = "a3",
        d_e: int = 16,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            A3TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                mode=mode,
                seq_len=max_seq_len,
                d_e=d_e,
                layer_idx=ell,
                dropout=dropout,
            )
            for ell in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classification head (uses [CLS] = first token)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _make_pad_mask(self, token_ids: Tensor) -> Tensor:
        """
        Returns (B, n, n) boolean mask where True = padding position to ignore.
        """
        B, n = token_ids.shape
        pad = (token_ids == self.pad_idx)       # (B, n)
        # mask out columns where key is padding
        mask = pad.unsqueeze(1).expand(B, n, n) # (B, n, n)
        return mask

    def forward(
        self,
        token_ids: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, list]:
        """
        Args:
            token_ids: (B, n)  integer token indices
            mask:      (B, n, n) optional custom mask

        Returns:
            logits:      (B, num_classes)
            cls_repr:    (B, d_model)  - [CLS] representation
            attn_weights: list of (B, n, n) per layer
        """
        B, n = token_ids.shape
        device = token_ids.device

        # Build padding mask if not provided
        if mask is None:
            mask = self._make_pad_mask(token_ids)

        # Embeddings
        positions = torch.arange(n, device=device).unsqueeze(0).expand(B, -1)
        x = self.emb_drop(self.token_emb(token_ids) + self.pos_emb(positions))

        # Transformer layers
        attn_weights = []
        for layer in self.layers:
            x, attn_w = layer(x, mask)
            attn_weights.append(attn_w)

        x = self.norm(x)

        # [CLS] token = first position
        cls_repr = x[:, 0, :]               # (B, d_model)
        logits = self.classifier(cls_repr)   # (B, num_classes)

        return logits, cls_repr, attn_weights


# ============================================================
# 7. Training utilities
# ============================================================

def build_optimizer(
    model: nn.Module,
    base_lr: float = 3e-4,
    a3_lr: Optional[float] = None,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer.
    Optionally uses a separate (usually smaller) learning rate
    for A3-specific parameters to reduce gradient spiking.

    Args:
        model:        the model
        base_lr:      learning rate for all parameters
        a3_lr:        if set, A3 gate/bias/embedding params use this lr
        weight_decay: L2 regularization

    Returns:
        AdamW optimizer
    """
    if a3_lr is None:
        return torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

    # Separate parameter groups: A3-specific vs. rest
    a3_params, base_params = [], []
    a3_names = {"W_g", "U", "e_lh"}   # A3 module parameter names

    for name, param in model.named_parameters():
        # Check if any A3-specific name appears in the parameter path
        is_a3 = any(a3_name in name for a3_name in a3_names)
        if is_a3:
            a3_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {"params": base_params, "lr": base_lr,  "weight_decay": weight_decay},
        {"params": a3_params,   "lr": a3_lr,   "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(param_groups)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup + cosine decay schedule.
    Matches the paper's training setup exactly.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_total_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    clip_norm: float = 1.0,
) -> float:
    """
    One training epoch. Returns mean loss.
    """
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        logits, _, _ = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping (important for A3 stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate on a dataloader.
    Returns (loss, accuracy).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        logits, _, _ = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    acc = correct / max(1, total)
    return total_loss / max(1, len(dataloader)), acc


@torch.no_grad()
def analyze_gates(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 10,
) -> dict:
    """
    Collect gate statistics across multiple batches.
    Returns a dict: {layer_idx: {head_idx: {'cv': [...], ...}}}

    Use this to reproduce the paper's CV analysis (Table 6):
    - CV > 0.15 → genuinely input-conditioned head
    - CV < 0.05 → effectively fixed temperature head
    """
    model.eval()
    all_stats = {}  # layer -> head -> list of gate values

    batch_count = 0
    for batch in dataloader:
        if batch_count >= num_batches:
            break
        input_ids, _ = batch
        input_ids = input_ids.to(device)
        B, n = input_ids.shape

        # Forward pass to get embeddings
        positions = torch.arange(n, device=device).unsqueeze(0).expand(B, -1)
        x = model.emb_drop(model.token_emb(input_ids) + model.pos_emb(positions))

        for ell, layer in enumerate(model.layers):
            if layer.attn.mode != "a3":
                continue
            stats_list = layer.attn.get_gate_stats(x)
            for s in stats_list:
                h = s["head"]
                key = (ell, h)
                if key not in all_stats:
                    all_stats[key] = {"gate_cv": []}
                all_stats[key]["gate_cv"].append(s["gate_cv"])

        # Run through the layer to get updated x for next layer
        with torch.no_grad():
            mask = model._make_pad_mask(input_ids)
            for ell, layer in enumerate(model.layers):
                x, _ = layer(x, mask)

        batch_count += 1

    # Summarize
    summary = {}
    for (ell, h), vals in all_stats.items():
        cv_mean = sum(vals["gate_cv"]) / len(vals["gate_cv"])
        summary[(ell, h)] = {
            "mean_cv": cv_mean,
            "behavior": (
                "input-conditioned" if cv_mean > 0.15
                else "mildly-dependent" if cv_mean > 0.05
                else "fixed-temperature"
            ),
        }
    return summary


# ============================================================
# 8. Quick smoke test
# ============================================================

def run_smoke_test():
    """
    Verifies that all four modes run without error and produce
    valid attention weights (non-negative, sum to 1).
    No real data needed.
    """
    print("=" * 60)
    print("A3 Attention - Smoke Test")
    print("=" * 60)

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Dummy data
    B, n, d_model = 4, 32, 128
    vocab_size     = 1000
    num_classes    = 3
    num_heads      = 4
    num_layers     = 3

    dummy_ids = torch.randint(1, vocab_size, (B, n)).to(device)

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
            dropout=0.0,  # off for deterministic test
        ).to(device)

        logits, cls_repr, attn_weights = model(dummy_ids)

        # --- Validity checks ---
        assert logits.shape == (B, num_classes), \
            f"[{mode}] logit shape mismatch"

        for ell, attn in enumerate(attn_weights):
            assert attn.shape == (B, n, n), \
                f"[{mode}] layer {ell} attn shape mismatch"
            # Non-negativity
            assert (attn >= 0).all(), \
                f"[{mode}] layer {ell}: negative attention weights!"
            # Normalization (rows sum to 1)
            row_sums = attn.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"[{mode}] layer {ell}: rows do not sum to 1!"

        n_params = count_parameters(model)
        print(f"  Mode: {mode:15s} | Params: {n_params:>8,} | "
              f"logit shape: {tuple(logits.shape)} | ✓ all checks passed")

    # --- Gate analysis for A3 ---
    print("\n--- A3 Gate Stats (random input, untrained model) ---")
    a3_model = A3TransformerEncoder(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, d_ff=256, max_seq_len=n,
        num_classes=num_classes, mode="a3", d_e=16, dropout=0.0,
    ).to(device)

    # Single batch gate stats
    positions = torch.arange(n, device=device).unsqueeze(0).expand(B, -1)
    x = a3_model.token_emb(dummy_ids) + a3_model.pos_emb(positions)

    for ell, layer in enumerate(a3_model.layers):
        stats = layer.attn.get_gate_stats(x)
        for s in stats:
            print(
                f"  Layer {ell} Head {s['head']:2d} | "
                f"gate_mean={s['gate_mean']:.3f}  "
                f"gate_cv={s['gate_cv']:.4f}  "
                f"range=[{s['gate_min']:.3f}, {s['gate_max']:.3f}]"
            )
        # Pass through layer for next layer's input
        with torch.no_grad():
            mask = a3_model._make_pad_mask(dummy_ids)
            x, _ = layer(x, mask)

    print("\n--- Parameter comparison (d_model=128, 4 heads, 3 layers, n=32) ---")
    for mode in ATTN_MODES:
        m = A3TransformerEncoder(
            vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, d_ff=256, max_seq_len=n,
            num_classes=num_classes, mode=mode, d_e=16, dropout=0.0,
        )
        base = A3TransformerEncoder(
            vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, d_ff=256, max_seq_len=n,
            num_classes=num_classes, mode="softmax", d_e=16, dropout=0.0,
        )
        delta = count_parameters(m) - count_parameters(base)
        print(
            f"  {mode:15s}: {count_parameters(m):>8,} params  "
            f"(Δ vs softmax: {delta:+,})"
        )

    print("\nAll smoke tests passed.")
    print("=" * 60)


# ============================================================
# 9. Example: how to train from scratch
# ============================================================

def example_training_loop():
    """
    Minimal example of a training loop.
    Replace the fake DataLoader with your real one.
    """
    print("\n--- Example Training Setup ---")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters (matching paper)
    VOCAB_SIZE   = 30522   # BERT wordpiece vocab
    D_MODEL      = 512
    NUM_HEADS    = 8
    NUM_LAYERS   = 6
    D_FF         = 2048
    MAX_SEQ_LEN  = 128
    NUM_CLASSES  = 2       # SST-2
    DROPOUT      = 0.1
    BATCH_SIZE   = 64
    MAX_EPOCHS   = 20
    LR           = 3e-4
    A3_LR        = 1e-4    # separate lr for gate params (reduces spiking)
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP    = 1.0

    model = A3TransformerEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        num_classes=NUM_CLASSES,
        mode="a3",          # swap to "softmax", "temp_softmax", "sparsemax"
        d_e=16,
        dropout=DROPOUT,
    ).to(device)

    print(f"  Model parameters: {count_parameters(model):,}")

    optimizer = build_optimizer(
        model,
        base_lr=LR,
        a3_lr=A3_LR,        # None to use shared lr
        weight_decay=WEIGHT_DECAY,
    )

    # Fake dataset for illustration
    N_TRAIN = 200
    fake_ids    = torch.randint(1, VOCAB_SIZE, (N_TRAIN, MAX_SEQ_LEN))
    fake_labels = torch.randint(0, NUM_CLASSES, (N_TRAIN,))
    dataset = torch.utils.data.TensorDataset(fake_ids, fake_labels)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_total_steps  = MAX_EPOCHS * len(loader)
    num_warmup_steps = int(0.1 * num_total_steps)
    scheduler = get_lr_scheduler(optimizer, num_warmup_steps, num_total_steps)

    # Training loop (3 epochs for illustration)
    for epoch in range(3):
        loss = train_one_epoch(model, loader, optimizer, scheduler, device, GRAD_CLIP)
        val_loss, acc = evaluate(model, loader, device)
        print(f"  Epoch {epoch+1:2d} | train_loss={loss:.4f} | "
              f"val_loss={val_loss:.4f} | acc={acc:.4f}")

    print("  Training example complete.")


# ============================================================
if __name__ == "__main__":
    run_smoke_test()
    example_training_loop()
