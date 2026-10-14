# Tapetum Framework

Research code and experiment artifacts for **Retinex-Tapetum**, a bio-inspired darkness-aware Retinex method for low-light image enhancement, together with multi-model benchmark workflows.

## Reproduce the experiments

The experiments were run in Google Colab using the publicly available notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/human-centered-computing/tapetum-framework/blob/main/TAPETUM/RetinexTapetum_ALL_models.ipynb)

- **[Colab notebook source](./TAPETUM/RetinexTapetum_ALL_models.ipynb)**
- **[Full TAPETUM README: setup, datasets, training, HPO, evaluation and checkpoints](./TAPETUM/README.md)**
- **[Implementation](./TAPETUM/RetinexTapetum)**
- **[Hyperparameter-search records](./TAPETUM/HyperparameterSearch)**
- **[Inference-speed measurement logs](./TAPETUM/SpeedMetrics)**
- **[Shared TAPETUM Google Drive workspace](https://drive.google.com/drive/folders/13ayyEC3V1wWdX3AXdfL8y7VqnL8eTPFT?usp=sharing)**

The workflow expects supporting data under `/content/drive/MyDrive/TAPETUM`. GitHub contains executable notebooks and source code, while some large datasets, checkpoints and baseline assets are Drive-backed. Consult the [detailed reproduction instructions](./TAPETUM/README.md#quick-start) before running; the same published results require the corresponding data, configuration and environment. The repository does not guarantee identical numerical results from an empty Colab session.

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](./LICENSE). Contributions are accepted under the same license.
