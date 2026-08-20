# LAMBDA_MAX Boundary-Refinement Study for LOL-v2 Real_captured

Generated: 2026-08-10T12:00:09+00:00

## Motivation

The dataset analysis measured a median target-to-input brightness gain of 7.7078 and a mean dark-pixel ratio of 0.9324. These values indicate a severe illumination-enhancement requirement. The preceding illumination search selected LAMBDA_MAX=2.0, exactly at the upper limit of its search interval. Therefore, a one-dimensional boundary-refinement study was performed to determine whether 2.0 was a genuine optimum or a boundary-censored selection.

## Controlled experimental design

- Dataset: LOL-v2 / Real_captured.
- Candidates: 2.0, 2.1, 2.2, 2.3, 2.4.
- Common source checkpoint: optimizer-stage cumulative epoch 25.
- Evaluation target: cumulative epoch 30; only 5 new epochs per candidate.
- Cosine scheduler horizon: 120 epochs.
- Adam optimizer moments and scheduler state were preserved.
- Stage-local validation history was reset so earlier-stage best scores could not dominate selection.
- Random seed and split seed: 42; validation ratio: 0.10.
- Exact main-HPO split size: 620 training and 69 validation images.
- Only LAMBDA_MAX changed; all other model, optimizer, and loss settings were held fixed.
- Validation images were not used to construct the candidate interval.

## Fixed hyperparameters

- BASE_CHANNELS: 128
- BATCH_SIZE: 2
- CROP_SIZE: 256
- GRAD_CLIP_NORM: 3.0
- LR: 0.0006620890956705852
- MIN_LR: 4.082960740748955e-05
- W_ATTN: 0.012
- W_CHROMA: 0.12
- W_COLOR: 0.06
- W_DARK_NOISE: 0.01
- W_EDGE: 0.04
- W_L1: 1.0
- W_LPIPS: 0.08
- W_RECON_HIGH: 0.85
- W_RECON_LOW: 1.0
- W_REFLECT: 0.06
- W_SMOOTH_ENH: 0.06
- W_SMOOTH_HIGH: 0.08
- W_SMOOTH_LOW: 0.08
- W_SSIM: 0.35

## Selection rule

Candidates within 0.15 dB of the maximum PSNR were retained. Among them, candidates within 0.002 of the maximum retained SSIM were kept. The candidate with the minimum LPIPS was then selected, with PSNR, SSIM, and the smaller LAMBDA_MAX used only as deterministic tie-breakers.

## Results

| LAMBDA_MAX | PSNR (dB) | SSIM | LPIPS | Best epoch | Highlight clip ratio | Luminance bias |
|---:|---:|---:|---:|---:|---:|---:|
| 2.00 | 20.8029 | 0.810681 | 0.143911 | 30 | 0.004748 | -0.012365 |
| 2.10 | 20.7459 | 0.804328 | 0.142542 | 27 | 0.006076 | 0.001746 |
| 2.20 | 20.8717 | 0.810071 | 0.144761 | 30 | 0.005487 | -0.006642 |
| 2.30 | 20.8305 | 0.809063 | 0.144510 | 30 | 0.005781 | -0.004135 |
| 2.40 | 20.8500 | 0.809297 | 0.144616 | 30 | 0.006067 | -0.001023 |

## Conclusion and required next step

The original upper-bound value, LAMBDA_MAX=2.0, remained optimal under the predefined multi-metric rule. The original fidelity and detail-stage results may therefore be retained, and final confirmation can proceed.

## Reproducibility safeguards

The implementation stores a result JSON, a resumable last checkpoint, a best checkpoint, a CSV summary, and this report under `HyperparameterSearch/lol_v2_real/lambda_boundary`. Interrupted runs resume from their latest cumulative epoch rather than restarting from epoch 25.
