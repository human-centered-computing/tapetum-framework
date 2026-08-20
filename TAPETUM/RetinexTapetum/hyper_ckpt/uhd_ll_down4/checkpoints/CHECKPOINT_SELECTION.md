# Checkpoint Selection — UHD-LL down4

## Selection rule

1. Evaluate every candidate using seeds `42`, `123`, and `3407`.
2. Compute the mean PSNR, SSIM, and LPIPS for each candidate.
3. Keep candidates within `0.15 dB` of the highest mean PSNR.
4. Among them, keep candidates within `0.002` of the highest mean SSIM.
5. Select the remaining candidate with the lowest mean LPIPS.
6. Apply the same PSNR → SSIM → LPIPS hierarchy to the selected candidate's seed runs to choose the deployment checkpoint.
7. Use `best.pth` from the recorded checkpoint epoch; do not use `last.pth`.

## Selected candidate

- Candidate: `source_profile_baseline`
- Three-seed mean: PSNR `27.0354`, SSIM `0.9263`, LPIPS `0.0670`

## Selected checkpoint

- Seed: `3407`
- Checkpoint epoch: `96`
- PSNR: `27.2593`
- SSIM: `0.9281`
- LPIPS: `0.0652`
- Deployment path: `checkpoints/selected/best.pth`
- Supporting files: `checkpoints/selected/result.json`, `checkpoints/selected/config_used.py`
