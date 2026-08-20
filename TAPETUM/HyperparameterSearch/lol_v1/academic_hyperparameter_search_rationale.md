# Dataset-Adaptive Hyperparameter Search Rationale

**Method:** RetinexTapetum dataset-specific HPO  
**Dataset:** LOL-v1  
**Dataset key:** `lol_v1`  
**Generated:** 2026-08-11T13:37:15

## 1. Purpose

This report documents how measurable properties of the selected paired low-light dataset were used to define the hyperparameter search space. Dataset analysis did not directly assign the final hyperparameter values. It determined which parameters were eligible for optimization and constrained their candidate domains; the final values were selected from validation metrics.

## 2. Analysis protocol and leakage control

The paired Train folders contained 485 matched Low/Normal images. The fixed split assigned 437 pairs to optimization and 48 pairs to validation. A deterministic, filename-distributed sample of 256 pairs (maximum 256) was analyzed using thumbnails bounded to 256 pixels.

The split used seed 42, a validation-ratio target of 0.1000, a validation-count cap of None, and a numeric filename block size of None. A block size other than `None` kept each numeric ID block wholly in one partition to reduce potential leakage between closely indexed captures.

Only the training partition was used to construct the adaptive search space. Validation targets were reserved for candidate comparison, and the Test partition was excluded from both search-space construction and hyperparameter selection. This separation reduces test leakage and preserves the Test partition for the final, post-HPO evaluation.

## 3. Measured dataset characteristics

| Characteristic | Statistic | Value | p10 | p90 |
|---|---:|---:|---:|---:|
| Target/input luminance gain | median | 8.8328 | 4.6291 | 20.5533 |
| Dark-pixel ratio (Y < 0.15) | mean | 0.9418 | 0.8094 | 1.0000 |
| Shadow-pixel ratio (Y < 0.25) | mean | 0.9835 | 0.9873 | 1.0000 |
| Dark-region noise proxy | mean | 0.0028 | 0.0008 | 0.0053 |
| Normalized color-balance shift | mean | 0.0992 | 0.0137 | 0.2475 |
| Target chroma strength | mean | 0.0363 | 0.0163 | 0.0620 |
| Target/input detail gain | median | 4.2970 | 2.6821 | 7.6255 |
| Target edge strength | mean | 0.0232 | 0.0129 | 0.0356 |
| Spatial darkness variation | mean | 0.0251 | 0.0096 | 0.0437 |
| Target highlight clipping ratio | mean | 0.0050 | 0.0000 | 0.0148 |
| Pair-size mismatch ratio | mean | 0.0000 | 0.0000 | 0.0000 |

### Operational definitions

- Luminance was computed as `Y = 0.299R + 0.587G + 0.114B`.
- Brightness gain was the ratio of mean target luminance to mean low-light luminance.
- The dark-region noise proxy was the mean absolute high-frequency residual between the low-light RGB image and a Gaussian-smoothed image, restricted to pixels with `Y < 0.25`. It is a proxy, not a calibrated sensor-noise estimate.
- Color shift was the mean absolute difference between normalized RGB channel means of paired Low and Normal images.
- Detail gain was the ratio of target to input mean gradient strength.
- Spatial darkness variation was the standard deviation of tile-level luminance means.

## 4. Prespecified decision rules

| Decision | Observed evidence | Prespecified rule | Outcome | Effect |
|---|---|---|---:|---|
| Severe enhancement | gain=8.8328; dark ratio=0.9418 | gain >= 4.00 OR dark ratio >= 0.60 | activated | Controls the LR interval and gradient-clipping candidates. |
| Large training set | training pairs=437 | training pairs >= 4000 | not activated | Selects a more conservative LR interval when activated. |
| Noisy low-light data | noise proxy=0.0028 | noise proxy >= 0.018 | not activated | Expands conservative optimization and noise-penalty ranges. |
| Structural fidelity demand | detail gain=4.2970; target edge=0.0232 | detail gain >= 1.10 OR target edge >= 0.035 | activated | Controls the W_SSIM and W_LPIPS domains. |
| Color/chroma search | color shift=0.0992; chroma=0.0363 | color shift >= 0.025 OR target chroma >= 0.075 | activated | Activates W_COLOR and W_CHROMA. |
| Edge-loss search | detail gain=4.2970; target edge=0.0232 | detail gain >= 1.05 OR target edge >= 0.030 | activated | Activates W_EDGE. |
| Dark-noise-loss search | noise proxy=0.0028; dark ratio=0.9418 | noise proxy >= 0.012 OR dark ratio >= 0.40 | activated | Activates W_DARK_NOISE. |
| Attention-regularization search | dark ratio=0.9418; spatial variation=0.0251 | dark ratio >= 0.35 OR spatial variation >= 0.08 | activated | Activates W_ATTN. |

## 5. Parameters admitted to optimization

| Stage | Parameter | Source-profile baseline | Candidate domain | Rationale |
|---|---|---:|---|---|
| optimizer | `LR` | 0.00046948014087030786 | 0.00012 to 0.0007 (log-uniform) | The dataset requires strong enhancement or contains substantial dark-region activity. The range covers conservative updates and the known source-profile baseline. |
| optimizer | `MIN_LR_RATIO` | 0.04533234500703289 | 0.02 to 0.15 (log-uniform) | The minimum learning rate is searched as a ratio of LR so the cosine schedule remains internally consistent. |
| optimizer | `GRAD_CLIP_NORM` | 3.0 | 1.0, 3.0, 5.0 | Gradient clipping candidates are selected from the observed enhancement severity to control unstable updates. |
| illumination | `LAMBDA_MAX` | 1.35 | 1.25 to 2 (uniform), step=0.05 | The normal-light targets are much brighter than the inputs; the illumination-amplification ceiling therefore needs a strong range. |
| fidelity | `W_SSIM` | 0.35 | 0.3 to 0.75 (uniform), step=0.05 | The SSIM-loss range is raised because target images show substantial structural/detail demand. |
| fidelity | `W_LPIPS` | 0.08 | 0.02 to 0.12 (uniform), step=0.01 | Perceptual weight is searched independently because LPIPS is also measured by a separate metric network. |
| detail | `W_COLOR` | 0.06 | 0.02 to 0.14 (uniform), step=0.02 | Observed channel-balance shift or target chroma justifies searching global color consistency. |
| detail | `W_CHROMA` | 0.12 | 0.04 to 0.18 (uniform), step=0.02 | Observed color/chroma demand justifies searching luminance-normalized chroma consistency. |
| detail | `W_EDGE` | 0.04 | 0 to 0.08 (uniform), step=0.01 | The normal-light targets contain more or stronger edges than the low-light inputs. |
| detail | `W_DARK_NOISE` | 0.01 | 0 to 0.05 (uniform), step=0.01 | Dark-region coverage or high-frequency activity justifies searching the chroma-noise penalty. |
| detail | `W_ATTN` | 0.012 | 0.001 to 0.012 (uniform), step=0.001 | The amount or spatial variation of darkness justifies tuning attention regularization. |

The source-profile value of every parameter admitted to a stage was included in that parameter's candidate domain, and the first-stage baseline trial represented the complete current source profile. Therefore, a prespecified generic boundary could not exclude the established value of an individual parameter.

## 6. Parameters held fixed

| Parameter | Fixed value | Reason for exclusion from HPO |
|---|---:|---|
| `BASE_CHANNELS` | 128 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `BATCH_SIZE` | 2 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `CROP_SIZE` | 256 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `W_L1` | 1.0 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `W_RECON_HIGH` | 0.85 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `W_RECON_LOW` | 1.0 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `W_REFLECT` | 0.06 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `W_SMOOTH_ENH` | 0.06 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `W_SMOOTH_HIGH` | 0.08 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |
| `W_SMOOTH_LOW` | 0.08 | Prespecified control retained to preserve the target architecture and the controlled training/loss design. |

## 7. Staged search budget and progressive continuation

| Stage | Completed trials targeted | Cumulative epoch target | Resume epoch | New epochs per trial |
|---|---:|---:|---:|---:|
| optimizer | 10 | 25 | 0 | 25 |
| illumination | 10 | 30 | 25 | 5 |
| fidelity | 12 | 35 | 30 | 5 |
| detail | 14 | 40 | 35 | 5 |

Epoch-budget rationale: The cumulative epoch targets were prespecified for this dataset profile before validation-based candidate comparison.

The planned search requires 430 trial-epochs when all targeted trials complete. After each stage, every candidate in the next stage branches from the selected trial's last checkpoint. Model weights, Adam state, scheduler state, and the completed-epoch index are preserved; only the additional epochs are executed. Previous best-metric history is reset at a stage boundary so that a newly changed hyperparameter is evaluated only on epochs produced under that stage's configuration.

## 8. Candidate evaluation and selection

Each candidate was evaluated on the same fixed validation split using the checkpoint with the highest validation PSNR. Selection used a lexicographic tolerance rule rather than an arbitrary weighted sum: candidates within 0.15 dB of the highest PSNR were retained; among them, candidates within 0.002 of the highest SSIM were retained; the candidate with the lowest LPIPS was then selected.

When optional final multi-seed confirmation is enabled, the exact source-profile baseline is included as a control configuration. This prevents the procedure from reporting an HPO configuration as an improvement merely because it was the least poor member of an HPO-only candidate set.

The search used seed 42, while the data split used seed 42. Optional final confirmation evaluates the exact source-profile baseline and the top 2 distinct HPO configurations using independent seeds [42, 123, 3407] for up to 120 epochs. Final multi-seed runs start independently rather than inheriting a search checkpoint, preserving seed-level independence.

## 9. Manuscript-ready methodological statement

For LOL-v1, the hyperparameter search space was constructed from prespecified rules applied exclusively to the training partition. The analysis measured target-to-input luminance gain (8.8328), dark-pixel coverage (0.9418), a dark-region high-frequency noise proxy (0.0028), normalized color shift (0.0992), detail gain (4.2970), target edge strength (0.0232), and spatial darkness variation (0.0251). These measurements activated dataset-relevant parameter groups and constrained their candidate domains, while architecture-defining and controlled loss parameters remained fixed at the source-profile values. Candidate configurations were compared on a fixed validation split using PSNR, SSIM, and LPIPS, with the Test partition reserved for post-selection evaluation.

## 10. Scope and limitations

The thresholds in this procedure are prespecified engineering rules, not learned decision boundaries and not statistical significance thresholds. The noise measurement is an image-domain proxy and does not replace calibrated sensor-noise estimation. The report should therefore be presented as a reproducible, data-informed search-space design strategy, not as proof that the thresholds are universally optimal.
