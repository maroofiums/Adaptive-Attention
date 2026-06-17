# Contributing

Thanks for your interest in this project. This started as a small, honestly-reported
empirical study, and contributions that maintain that spirit are especially welcome.

## Ways to contribute

- **Large-scale evaluation.** The biggest open question in the paper is whether
  results hold at BERT-base/GPT-2 scale with real pre-training. If you have the
  compute to test this, please share results (positive, negative, or mixed).
- **Matched-parameter baselines.** Help disentangle "more parameters" from "the
  specific A³ mechanism" by adding a controlled baseline (e.g., equivalent
  parameters added to feed-forward layers instead).
- **Additional baselines.** An α-entmax integration (via the
  [`entmax`](https://github.com/deep-spin/entmax) package) would strengthen the
  comparison table.
- **Additional tasks.** Machine translation, long-context modeling, and generation
  are not covered in the current study.
- **Bug fixes / code quality.** Standard PRs welcome - please include or update
  tests in `tests/test_a3_attention.py`.

## Before submitting a PR

1. Open an issue first for anything beyond a small fix, so we can discuss scope.
2. Run the test suite locally: `pytest tests/ -v`
3. Run the smoke test: `python scripts/run_smoke_test.py`
4. If you add a new attention mode or module, add corresponding tests that check:
   - output shapes
   - validity (non-negativity, row-sums-to-1) where applicable
   - gradient flow (no NaNs)

## Reporting results

If you run additional experiments (especially at larger scale), please report:
- Number of seeds and the statistical method used (the paper uses paired bootstrap
  with 10,000 resamples)
- Full mean ± std, not just best-seed numbers
- Negative or null results are valuable and explicitly welcome - the spirit of this
  repository is honest reporting over flattering numbers.

## Code style

- Type hints on public functions where practical
- Docstrings explaining *why*, not just *what*, especially for anything tied to a
  proposition or claim in the paper
- Keep new dependencies minimal; prefer stdlib/PyTorch-only where possible
