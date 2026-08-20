# Checkpoint Selection — LOL-v2 Real-Captured

## Selection rule

1. Evaluate every candidate using seeds `42`, `123`, and `3407`.
2. Compute the mean PSNR, SSIM, and LPIPS for each candidate.
3. Keep candidates within `0.15 dB` of the highest mean PSNR.
4. Among them, keep candidates within `0.002` of the highest mean SSIM.
5. Select the remaining candidate with the lowest mean LPIPS.
6. Apply the same PSNR → SSIM → LPIPS hierarchy to the selected candidate's seed runs to choose the deployment checkpoint.
7. Use `best.pth` from the recorded checkpoint epoch; do not use `last.pth`.

## Selected candidate

- Candidate: `hpo_candidate_02_trial_0012_3df5d5c6d5`
- Three-seed mean: PSNR `21.5759`, SSIM `0.8317`, LPIPS `0.1615`

## Selected checkpoint

- Seed: `123`
- Checkpoint epoch: `74`
- PSNR: `21.5976`
- SSIM: `0.8318`
- LPIPS: `0.1532`
- Deployment path: `checkpoints/selected/best.pth`
- Supporting files: `checkpoints/selected/result.json`, `checkpoints/selected/config_used.py`
