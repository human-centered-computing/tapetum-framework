# Tapetum Framework

**Human-Centered AI · Inspectable Computer Vision · Reproducible Low-Light Image Enhancement**

This repository documents **Retinex-Tapetum**, a bio-inspired, darkness-aware Retinex research framework. The work is presented as part of a **human-centered and explainability-oriented AI research direction**: making image-enhancement decisions easier for researchers and practitioners to inspect, diagnose, reproduce, and critically evaluate.

The approach exposes intermediate representations such as reflectance-like and illumination-like maps, a darkness prior, a learned Tapetum Attention Map, and a Darkness-Gated Spatial Amplification Map. These offer *model inspectability* and support diagnostic interventions; they are **not** independently validated causal explanations or a substitute for user-centered studies.

## Visual overview

**Four-stage processing architecture**

![Retinex-Tapetum processing architecture from decomposition and guidance to illumination modulation and bounded residual refinement](TAPETUM/paper_retinextapetum/figures/tapetum_architecture_diagram.png)

**Low-light to enhanced-image illustration**

![Retinex-Tapetum visual low-light enhancement illustration](TAPETUM/paper_retinextapetum/figures/from_real_to_enhancement_retinex_tapetum.png)

The images are illustrative research artifacts; qualitative examples do not establish consistent image quality or a human preference advantage on their own. See the [annotated figure gallery](./TAPETUM/paper_retinextapetum/figures/README.md) for the full set, including the tapetum lucidum inspiration illustration.

## Research focus

- **Human-oriented visual quality:** improving low-light image legibility while assessing fidelity, structural quality, and perceptual differences.
- **Inspectable intermediate representations:** exposing learned guidance and amplification maps rather than describing the enhancer solely as a black box.
- **Reproducible research:** publicly accessible Colab workflows, dataset split manifests, HPO records, and benchmark diagnostics.
- **Responsible interpretation:** treating attention and learned maps as diagnostic signals, not as proof of feature-level causal explanations.

## Reproduce the experiments

The research experiments were run using the publicly accessible Google Colab workflow:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/human-centered-computing/tapetum-framework/blob/main/TAPETUM/RetinexTapetum_ALL_models.ipynb)

- **[Colab notebook source](./TAPETUM/RetinexTapetum_ALL_models.ipynb)**
- **[Full TAPETUM README: method, setup, datasets, HPO, evaluation, and checkpoints](./TAPETUM/README.md)**
- **[Annotated figure gallery](./TAPETUM/paper_retinextapetum/figures/README.md)**
- **[Implementation](./TAPETUM/RetinexTapetum)**
- **[Hyperparameter search records](./TAPETUM/HyperparameterSearch)**
- **[Inference speed logs](./TAPETUM/SpeedMetrics)**
- **[Shared TAPETUM Google Drive workspace](https://drive.google.com/drive/folders/13ayyEC3V1wWdX3AXdfL8y7VqnL8eTPFT?usp=sharing)**

The workflow expects supporting files at `/content/drive/MyDrive/TAPETUM`. Some large datasets, baselines, checkpoints, and experiment artifacts are Drive-backed rather than committed to GitHub. Follow the [reproduction instructions](./TAPETUM/README.md#quick-start); an empty Colab session does not by itself guarantee identical numerical results.

## Scope and research limitations

Retinex-Tapetum is motivated by the functional light-reuse analogy of the *tapetum lucidum*, not a physically faithful simulation of the eye. The provided diagnostic maps are model-generated representations, and fixed-checkpoint interventions should not be interpreted as evidence from retrained architectural ablations. Formal explanation-faithfulness tests and studies with end users remain opportunities for future research.

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](./LICENSE). Contributions are accepted under the same license.
