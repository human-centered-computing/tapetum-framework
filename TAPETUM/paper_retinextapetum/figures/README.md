# Retinex-Tapetum — Research Figure Gallery

These figures are existing research artifacts associated with **Retinex-Tapetum**, an inspectable, darkness-aware Retinex approach for low-light image enhancement. The gallery supports a human-centered / explainability-oriented research presentation by showing the biological motivation, model architecture, and visual enhancement examples without implying that a user study or formal explanation-fidelity validation has been conducted.

[← TAPETUM README](../../README.md) · [Repository homepage](../../../README.md) · [Open Colab notebook](https://colab.research.google.com/github/human-centered-computing/tapetum-framework/blob/main/TAPETUM/RetinexTapetum_ALL_models.ipynb)

## 1. Four-stage architecture and inspectable representations

![Four-stage Retinex-Tapetum architecture, including Retinex decomposition, auxiliary guidance, darkness-gated illumination enhancement, and bounded residual refinement](./tapetum_architecture_diagram.png)

**Figure file:** [`tapetum_architecture_diagram.png`](./tapetum_architecture_diagram.png)

The model explicitly separates Retinex decomposition, auxiliary cues, Tapetum-guided illumination enhancement, and reconstruction plus bounded residual refinement. Key intermediate values are `R_low`, `L_low`, `D`, `T`, `Lambda`, `Y_HF`, and `Delta I`. This visibility aids **model inspection**, debugging, and hypothesis-driven diagnostic analysis. Such maps are learned signals, not measured causal feature-attributions or direct physical quantities.

## 2. Low-light enhancement illustration

![Retinex-Tapetum qualitative low-light image enhancement illustration](./from_real_to_enhancement_retinex_tapetum.png)

**Figure file:** [`from_real_to_enhancement_retinex_tapetum.png`](./from_real_to_enhancement_retinex_tapetum.png)

A qualitative illustration associated with Retinex-Tapetum low-light image enhancement. Treat the image as a visual research example, **not** as evidence of a universally better result or a controlled human preference evaluation. Quantitative performance should be interpreted using the independent test metrics and evaluation protocol in the manuscript and project README.

## 3. Bio-inspired motivation: tapetum lucidum

![Tapetum lucidum biological inspiration illustration used with Retinex-Tapetum](./tapedum_lucidum.png)

**Figure file:** [`tapedum_lucidum.png`](./tapedum_lucidum.png)

The biological *tapetum lucidum* motivates a **functional light-reuse analogy**. The model operates entirely in the image domain and does not reconstruct animal-eye anatomy, simulate optical photon transport, or claim physiological equivalence. The filename `tapedum_lucidum.png` is retained exactly as stored in the repository.

## Human-centered and explainability-oriented interpretation

The repository contributes an **inspectable workflow for vision research**: interpretable-by-design *structure* (explicit model stages), accessible intermediate *signals* (learned maps and cues), reproducible *experiments* (Colab, split manifests, and HPO records), and *diagnostic interventions* on existing checkpoints. These support transparency and critical examination of the enhancement behavior.

**Important limits:** No end-user human-subject evaluation, validated explanation-faithfulness study, or causal attribution guarantee is claimed here. Fixed-weight interventions do not substitute for training new ablated models. The artifacts should be used with appropriate dataset licenses and the project license.

## Reproduction and citation

See [the TAPETUM reproducibility instructions](../../README.md#quick-start), the [training and HPO artifacts](../../HyperparameterSearch), and the [manuscript information](../../README.md#paper). Images should be cited as project research figures; they are not a substitute for reading the experimental protocol.
