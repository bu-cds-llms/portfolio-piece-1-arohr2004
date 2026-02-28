# Scaled Dot-Product Attention From Scratch: Why `1/sqrt(d_k)` Stabilizes Transformer Attention

This project is a portfolio mini-report for DS 593 focused on implementing and analyzing attention from first principles in PyTorch. The central question is why Transformers scale attention logits by 1/sqrt(d_k) and how that changes behavior as key/query dimension grows.

## Overview

The notebook builds scaled dot-product attention step by step, verifies correctness on small and larger matrices, and runs a  sweep over d_k = {4, 16, 64, 256} under scaled and unscaled conditions.  
Using ideas such as, logit growth, softmax saturation, and scaling to build empirical evidence using metrics, checks, and visualizations.

## Methods

The main analysis is in:

- `Portifilo_Piece_1.ipynb`

Core components:

- From-scratch implementation of scaled dot-product attention
- Numerical stabilization before softmax
- Sanity checks on small tensors and a larger matrix test case
- Masking edge-case sanity check (masked positions get near-zero attention)
- Sweep across d_k for scaled vs unscaled attention
- Metrics:
  - mean max attention weight (peakedness)
  - mean attention entropy (spread)
  - gradient norms for `Q` and `K`
- Heatmaps for visual comparison across dimensions
- Multi-seed robustness summary (mean ± std across seeds)
- Parity check against `torch.nn.functional.scaled_dot_product_attention`

Why these choices:

- Controlled d_k sweeps isolate dimensionality effects predicted by theory.
- Max-weight + entropy quantify attention concentration and spread.
- Gradient norms connect attention behavior to optimization stability.
- PyTorch check validates implementation correctness.

## Key Results

From `outputs/attention_sweep_results.csv`:

- As `d_k` increases, **unscaled** attention becomes much sharper.
- At `d_k=256`:
  - unscaled `mean_max_weight = 0.9942`
  - scaled `mean_max_weight = 0.3680`
  - unscaled `mean_entropy = 0.0296`
  - scaled `mean_entropy = 1.7256`
-Scaling by sqrt(d_k) prevents softmax saturation and keeps attention distributions more stable across dimensions.
- Multi-seed robustness (`outputs/attention_robustness_summary.csv`) shows the same scaled-vs-unscaled trend continues across seeds.
- The parity check reports a very small max absolute difference relative to PyTorch's built-in attention. 


## Repository Structure

- `Portifilo_Piece_1.ipynb`: main report notebook
- `requirements.txt`: project dependencies
- `outputs/attention_sweep_results.csv`: metric table from the sweep
- `outputs/attention_robustness_summary.csv`: multi-seed mean/std summary
- `outputs/attention_heatmap_gallery.png`: scaled vs unscaled heatmaps across d_k
- `outputs/gradient_norms_vs_dk.png`: gradient-norm comparison plot

## How To Run

1. Clone the repository and open it in VS Code (or Jupyter).
2. Create and activate an environment (recommended):
   - `python3 -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux)
3. Install dependencies:
   - `python -m pip install -r requirements.txt`
4. Open `Portifilo_Piece_1.ipynb`.
5. Select the same interpreter used for installation.
6. Run all cells from top to bottom.

The notebook will save outputs to the `outputs/` folder automatically.

## Requirements

See `requirements.txt`:

- `matplotlib>=3.7`
- `numpy>=1.24`
- `pandas>=2.0`
- `torch>=2.0`
- `ipykernel>=6.29`

## Notes and Limitations

- The experiment uses synthetic random `Q/K/V`, not a fully trained language model.
- Results are reproducible with fixed seeds but exact values may change with different seeds/hardware.
