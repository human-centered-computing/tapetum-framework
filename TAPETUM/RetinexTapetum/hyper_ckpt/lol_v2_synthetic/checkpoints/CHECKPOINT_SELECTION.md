# Checkpoint Selection — LOL-v2 Synthetic

## Selection rule

1. Evaluate every candidate using seeds `42`, `123`, and `3407`.
2. Compute the mean PSNR, SSIM, and LPIPS for each candidate.
3. Keep candidates within `0.15 dB` of the highest mean PSNR.
4. Among them, keep candidates within `0.002` of the highest mean SSIM.
5. Select the remaining candidate with the lowest mean LPIPS.
6. Apply the same PSNR → SSIM → LPIPS hierarchy to the selected candidate's seed runs to choose the deployment checkpoint.
7. Use `best.pth` from the recorded checkpoint epoch; do not use `last.pth`.

## Selected candidate

- Candidate: `hpo_candidate_01_trial_0007_f0fca1ff96`
- Three-seed mean: PSNR `25.1726`, SSIM `0.9357`, LPIPS `0.0554`

## Selected checkpoint

- Seed: `3407`
- Checkpoint epoch: `111`
- PSNR: `25.1167`
- SSIM: `0.9375`
- LPIPS: `0.0538`
- Deployment path: `checkpoints/selected/best.pth`
- Supporting files: `checkpoints/selected/result.json`, `checkpoints/selected/config_used.py`
