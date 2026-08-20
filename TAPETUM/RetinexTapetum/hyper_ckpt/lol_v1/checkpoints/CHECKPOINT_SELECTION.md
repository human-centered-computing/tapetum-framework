# Checkpoint Selection — LOL-v1

## Selection rule

1. Evaluate every candidate using seeds `42`, `123`, and `3407`.
2. Compute the mean PSNR, SSIM, and LPIPS for each candidate.
3. Keep candidates within `0.15 dB` of the highest mean PSNR.
4. Among them, keep candidates within `0.002` of the highest mean SSIM.
5. Select the remaining candidate with the lowest mean LPIPS.
6. Apply the same PSNR → SSIM → LPIPS hierarchy to the selected candidate's seed runs to choose the deployment checkpoint.
7. Use `best.pth` from the recorded checkpoint epoch; do not use `last.pth`.

## Selected candidate

- Candidate: `hpo_candidate_02_trial_0002_f1bfd5bf87`
- Three-seed mean: PSNR `21.9074`, SSIM `0.8652`, LPIPS `0.1256`

## Selected checkpoint

- Seed: `42`
- Checkpoint epoch: `88`
- PSNR: `22.0090`
- SSIM: `0.8681`
- LPIPS: `0.1185`
- Deployment path: `checkpoints/selected/best.pth`
- Supporting files: `checkpoints/selected/result.json`, `checkpoints/selected/config_used.py`
