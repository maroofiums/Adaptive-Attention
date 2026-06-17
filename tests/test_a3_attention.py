"""
Unit tests for A3 attention implementation.

Run with:
    pytest tests/ -v
"""
import math
import sys
import os

import pytest
import torch
import torch.nn as nn

# Allow importing from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from a3_attention import (
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
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def _seed():
    set_seed(0)


@pytest.fixture
def small_qkv():
    B, n, d_k = 4, 16, 32
    Q = torch.randn(B, n, d_k)
    K = torch.randn(B, n, d_k)
    V = torch.randn(B, n, d_k)
    return Q, K, V


# ============================================================
# Sparsemax
# ============================================================

class TestSparsemax:
    def test_output_is_valid_distribution(self):
        sp = Sparsemax()
        z = torch.randn(8, 20)
        p = sp(z)
        assert (p >= 0).all()
        assert torch.allclose(p.sum(-1), torch.ones(8), atol=1e-4)

    def test_produces_exact_zeros(self):
        """Sparsemax should produce some exact zeros, unlike softmax."""
        sp = Sparsemax()
        z = torch.tensor([[5.0, 0.1, 0.05, -3.0, -5.0]])
        p = sp(z)
        assert (p == 0).any(), "sparsemax should zero out low-scoring entries"

    def test_handles_batched_input(self):
        sp = Sparsemax()
        z = torch.randn(3, 5, 10)  # extra batch dim
        p = sp(z)
        assert p.shape == z.shape
        assert torch.allclose(p.sum(-1), torch.ones(3, 5), atol=1e-4)


# ============================================================
# Baseline attention modules
# ============================================================

class TestSoftmaxAttention:
    def test_valid_attention(self, small_qkv):
        Q, K, V = small_qkv
        attn_mod = SoftmaxAttention()
        out, attn = attn_mod(Q, K, V)
        B, n, d_k = Q.shape
        assert out.shape == (B, n, d_k)
        assert attn.shape == (B, n, n)
        assert (attn >= 0).all()
        assert torch.allclose(attn.sum(-1), torch.ones(B, n), atol=1e-4)

    def test_no_learnable_parameters(self):
        attn_mod = SoftmaxAttention()
        assert count_parameters(attn_mod) == 0

    def test_respects_mask(self, small_qkv):
        Q, K, V = small_qkv
        B, n, _ = Q.shape
        mask = torch.zeros(B, n, n, dtype=torch.bool)
        mask[:, :, -1] = True  # mask out last key position
        attn_mod = SoftmaxAttention()
        _, attn = attn_mod(Q, K, V, mask)
        assert torch.allclose(attn[:, :, -1], torch.zeros(B, n), atol=1e-6)


class TestTempSoftmaxAttention:
    def test_valid_attention(self, small_qkv):
        Q, K, V = small_qkv
        attn_mod = TempSoftmaxAttention()
        out, attn = attn_mod(Q, K, V)
        assert (attn >= 0).all()
        assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4)

    def test_one_parameter(self):
        attn_mod = TempSoftmaxAttention()
        assert count_parameters(attn_mod) == 1

    def test_tau_initialized_to_one(self):
        attn_mod = TempSoftmaxAttention()
        assert torch.isclose(attn_mod.tau, torch.tensor(1.0), atol=1e-6)

    def test_tau_always_positive(self):
        attn_mod = TempSoftmaxAttention()
        attn_mod.log_tau.data.fill_(-10.0)  # even with very negative log_tau
        assert attn_mod.tau.item() > 0


class TestSparsemaxAttention:
    def test_valid_attention(self, small_qkv):
        Q, K, V = small_qkv
        attn_mod = SparsemaxAttention()
        out, attn = attn_mod(Q, K, V)
        assert (attn >= 0).all()
        assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4)

    def test_no_learnable_parameters(self):
        attn_mod = SparsemaxAttention()
        assert count_parameters(attn_mod) == 0


# ============================================================
# A3Attention (core proposed module)
# ============================================================

class TestA3Attention:
    def test_valid_attention_random_init(self, small_qkv):
        """Validity (Proposition 1) must hold for arbitrary parameters."""
        Q, K, V = small_qkv
        B, n, d_k = Q.shape
        a3 = A3Attention(d_k=d_k, seq_len=n, d_e=8)
        # Perturb parameters away from the recovery initialization
        for p in a3.parameters():
            nn.init.normal_(p, std=2.0)

        out, attn = a3(Q, K, V)
        assert out.shape == (B, n, d_k)
        assert attn.shape == (B, n, n)
        assert (attn >= 0).all(), "A3 must never produce negative attention weights"
        assert torch.allclose(
            attn.sum(-1), torch.ones(B, n), atol=1e-4
        ), "A3 attention rows must sum to 1 for any parameter values"

    def test_softmax_recovery_at_init(self, small_qkv):
        """Proposition 2: default init should closely match standard softmax."""
        Q, K, V = small_qkv
        B, n, d_k = Q.shape

        a3 = A3Attention(d_k=d_k, seq_len=n, d_e=8, gate_init_bias=3.0)
        softmax_mod = SoftmaxAttention()

        _, attn_a3 = a3(Q, K, V)
        _, attn_softmax = softmax_mod(Q, K, V)

        # Should be close but not necessarily exact, since sigmoid(3) != 1
        assert torch.allclose(attn_a3, attn_softmax, atol=0.05), (
            "A3 at default init should closely approximate standard softmax"
        )

    def test_gate_near_one_at_init(self, small_qkv):
        Q, K, _ = small_qkv
        d_k = Q.size(-1)
        a3 = A3Attention(d_k=d_k, seq_len=Q.size(1), d_e=8, gate_init_bias=3.0)
        stats = a3.gate_stats(Q, K)
        assert stats["gate_mean"] > 0.9, "gate should start near 1.0 with default init"

    def test_gradients_flow(self, small_qkv):
        Q, K, V = small_qkv
        Q.requires_grad_(True)
        d_k = Q.size(-1)
        a3 = A3Attention(d_k=d_k, seq_len=Q.size(1), d_e=8)
        out, _ = a3(Q, K, V)
        loss = out.sum()
        loss.backward()

        assert Q.grad is not None
        assert not torch.isnan(Q.grad).any()
        for name, p in a3.named_parameters():
            assert p.grad is not None, f"no gradient for parameter {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    def test_gate_collapse_with_zero_init(self, small_qkv):
        """
        Regression test for Failure Mode 1 in the paper: initializing
        b_g = 0 (instead of 3) leads to gate ~ 0.5, which is the
        documented unstable configuration.
        """
        Q, K, _ = small_qkv
        d_k = Q.size(-1)
        a3 = A3Attention(d_k=d_k, seq_len=Q.size(1), d_e=8, gate_init_bias=0.0)
        stats = a3.gate_stats(Q, K)
        assert abs(stats["gate_mean"] - 0.5) < 0.05, (
            "gate_init_bias=0 should produce gate values near 0.5 (sigmoid(0))"
        )

    def test_parameter_count_matches_paper_estimate(self):
        """A3 should add a small number of parameters per head, not O(n^2)."""
        d_k, seq_len, d_e = 64, 128, 16
        a3 = A3Attention(d_k=d_k, seq_len=seq_len, d_e=d_e)
        n_params = count_parameters(a3)
        # d_ctx = 2*64+16=144; W_g: 144*64+64; U: 128*144; e_lh: 16
        expected = (144 * 64 + 64) + (128 * 144) + 16
        assert n_params == expected


# ============================================================
# MultiHeadAttention (all four modes)
# ============================================================

class TestMultiHeadAttention:
    @pytest.mark.parametrize("mode", ATTN_MODES)
    def test_all_modes_produce_valid_output(self, mode):
        B, n, d_model, H = 4, 16, 64, 4
        x = torch.randn(B, n, d_model)
        mha = MultiHeadAttention(d_model=d_model, num_heads=H, mode=mode, seq_len=n)
        out, attn = mha(x)

        assert out.shape == (B, n, d_model)
        assert attn.shape == (B, n, n)
        assert (attn >= 0).all(), f"mode={mode}: negative attention"
        assert torch.allclose(
            attn.sum(-1), torch.ones(B, n), atol=1e-4
        ), f"mode={mode}: rows do not sum to 1"

    def test_invalid_mode_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=64, num_heads=4, mode="not_a_real_mode", seq_len=16)

    def test_d_model_must_be_divisible_by_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4, mode="softmax", seq_len=16)

    def test_gate_stats_empty_for_non_a3_modes(self):
        x = torch.randn(2, 8, 32)
        mha = MultiHeadAttention(d_model=32, num_heads=2, mode="softmax", seq_len=8)
        assert mha.get_gate_stats(x) == []

    def test_gate_stats_populated_for_a3(self):
        x = torch.randn(2, 8, 32)
        mha = MultiHeadAttention(d_model=32, num_heads=2, mode="a3", seq_len=8, d_e=4)
        stats = mha.get_gate_stats(x)
        assert len(stats) == 2  # num_heads
        for s in stats:
            assert "gate_cv" in s


# ============================================================
# Transformer block and full encoder
# ============================================================

class TestA3TransformerBlock:
    @pytest.mark.parametrize("mode", ATTN_MODES)
    def test_residual_shapes_preserved(self, mode):
        B, n, d_model = 2, 12, 32
        block = A3TransformerBlock(
            d_model=d_model, num_heads=4, d_ff=64, mode=mode, seq_len=n, dropout=0.0
        )
        x = torch.randn(B, n, d_model)
        out, attn = block(x)
        assert out.shape == x.shape


class TestA3TransformerEncoder:
    @pytest.mark.parametrize("mode", ATTN_MODES)
    def test_forward_pass_shapes(self, mode):
        vocab_size, d_model, H, L, n, C = 500, 64, 4, 2, 16, 3
        model = A3TransformerEncoder(
            vocab_size=vocab_size, d_model=d_model, num_heads=H, num_layers=L,
            d_ff=128, max_seq_len=n, num_classes=C, mode=mode, dropout=0.0,
        )
        ids = torch.randint(1, vocab_size, (5, n))
        logits, cls_repr, attn_weights = model(ids)

        assert logits.shape == (5, C)
        assert cls_repr.shape == (5, d_model)
        assert len(attn_weights) == L
        for attn in attn_weights:
            assert (attn >= 0).all()
            assert torch.allclose(attn.sum(-1), torch.ones(5, n), atol=1e-4)

    def test_padding_mask_applied(self):
        vocab_size, d_model, H, L, n, C = 500, 32, 2, 2, 10, 2
        model = A3TransformerEncoder(
            vocab_size=vocab_size, d_model=d_model, num_heads=H, num_layers=L,
            d_ff=64, max_seq_len=n, num_classes=C, mode="softmax",
            dropout=0.0, pad_idx=0,
        )
        ids = torch.randint(1, vocab_size, (2, n))
        ids[:, -3:] = 0  # pad last 3 positions
        _, _, attn_weights = model(ids)
        for attn in attn_weights:
            # attention to padded key positions should be ~0
            assert torch.allclose(
                attn[:, :, -3:], torch.zeros_like(attn[:, :, -3:]), atol=1e-5
            )

    def test_a3_overhead_is_small_relative_to_softmax(self):
        """Sanity check that A3 overhead stays in the small-percentage range."""
        kwargs = dict(
            vocab_size=1000, d_model=128, num_heads=4, num_layers=3,
            d_ff=256, max_seq_len=32, num_classes=3, dropout=0.0,
        )
        base = count_parameters(A3TransformerEncoder(mode="softmax", **kwargs))
        a3 = count_parameters(A3TransformerEncoder(mode="a3", **kwargs))
        overhead_pct = 100 * (a3 - base) / base
        assert 0 < overhead_pct < 20, f"unexpected A3 overhead: {overhead_pct:.2f}%"


# ============================================================
# Optimizer / scheduler utilities
# ============================================================

class TestTrainingUtilities:
    def test_build_optimizer_shared_lr(self):
        model = A3TransformerEncoder(
            vocab_size=100, d_model=32, num_heads=2, num_layers=2,
            d_ff=64, max_seq_len=8, num_classes=2, mode="a3", dropout=0.0,
        )
        opt = build_optimizer(model, base_lr=1e-3)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == 1e-3

    def test_build_optimizer_separate_a3_lr(self):
        model = A3TransformerEncoder(
            vocab_size=100, d_model=32, num_heads=2, num_layers=2,
            d_ff=64, max_seq_len=8, num_classes=2, mode="a3", dropout=0.0,
        )
        opt = build_optimizer(model, base_lr=1e-3, a3_lr=1e-4)
        assert len(opt.param_groups) == 2
        lrs = sorted(g["lr"] for g in opt.param_groups)
        assert lrs == [1e-4, 1e-3]

    def test_lr_scheduler_warmup_then_decay(self):
        model = nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = get_lr_scheduler(opt, num_warmup_steps=10, num_total_steps=100)

        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        # LR should increase during warmup
        assert lrs[5] < lrs[9]
        # and should be near zero at the end (cosine decay)
        assert lrs[-1] < lrs[15]


# ============================================================
# Reproducibility
# ============================================================

class TestReproducibility:
    def test_set_seed_gives_deterministic_output(self):
        set_seed(123)
        a = torch.randn(10)
        set_seed(123)
        b = torch.randn(10)
        assert torch.equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
