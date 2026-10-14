# Tapetum Framework

**Human-Centered AI · Interpretable Intermediate Representations · Auditable Experimental Protocols**

**Research objective:** To advance **Human-Centered AI** by developing **interpretable intermediate representations and auditable experimental protocols** for low-light image enhancement.

**Türkçe araştırma odağı:** İnsan merkezli yapay zekâ (Human-Centered AI) hedefi doğrultusunda, düşük ışık görüntü iyileştirme için yorumlanabilir ara temsiller ve denetlenebilir deney protokolleri geliştirmeye yönelik bir araştırma.

This repository documents **Retinex-Tapetum**, a bio-inspired, darkness-aware Retinex framework that investigates how explicit enhancement stages, inspectable guidance maps, and reproducible experimental records can support more transparent development and critical assessment of computer-vision systems. The human-centered framing is a **research goal and design orientation**, not a claim of validated human-user benefit or formally proven explanations.

The approach exposes intermediate representations such as reflectance-like and illumination-like maps, a darkness prior, a learned Tapetum Attention Map, and a Darkness-Gated Spatial Amplification Map. These offer *model inspectability* and support diagnostic interventions; they are **not** independently validated causal explanations or a substitute for user-centered studies.

## Visual overview — from perception and camera imaging to bio-inspired enhancement

### 1. Retinex perception, camera acquisition, and the observed RGB image

<p align="center">
  <img src="TAPETUM/paper_retinextapetum/figures/from_real_to_enhancement_retinex_tapetum.png" alt="Conceptual relation between Retinex theories of human scene perception, camera image formation and processing, and the resulting RGB image with acquisition-related effects" width="960">
</p>

**Read the composite from left to right:**

1. **Retinex and human scene perception:** The first part introduces the idea that our perception of scene lightness and color depends on illumination, surface reflectance, and visual context, rather than being a direct copy of measured RGB values. In the simplified Retinex image model, illumination and reflectance are useful conceptual factors, not uniquely measured physical quantities.
2. **Camera image formation and processing:** The middle part represents how the physical scene becomes a camera signal through image acquisition and camera processing. Depending on the imaging system, this can include optics, exposure, sensing, color processing, and other image-signal-processing (ISP) steps.
3. **Observed RGB image:** The last part represents the resulting RGB image as a combination of scene-dependent information and camera-pipeline effects, which can include sensor noise, processing artifacts, and other acquisition distortions. Thus, recorded RGB pixels are **not** a direct, noise-free representation of the physical scene.

**Türkçe özet:** İlk bölüm Retinex'in insanın gerçek sahnede renk ve aydınlığı nasıl algıladığını açıklayan kuramsal yaklaşımını, orta bölüm fotoğraf makinesinin görüntü oluşturma ve işleme aşamalarını, son bölüm ise fiziksel sahnenin kamera işlemleri ve gürültü gibi etkilerle birlikte kaydedilmiş RGB görüntüye dönüşmesini anlatır.

### 2. Retinex and the biological tapetum lucidum

<p align="center">
  <img src="TAPETUM/paper_retinextapetum/figures/tapedum_lucidum.png" alt="Conceptual comparison of Retinex scene perception and the biological tapetum lucidum light-return mechanism that motivates Retinex-Tapetum" width="900">
</p>

The second illustration connects **two related but distinct ideas**:

- **Retinex:** a theory-inspired computational view of human lightness and color perception, using the relationship between scene illumination and reflectance.
- **Biological tapetum lucidum:** a reflective structure found behind the retina in many animals that returns some incident light through the photoreceptor layer, providing another opportunity for light detection in dim conditions.

**Retinex-Tapetum bridges these ideas computationally:** It learns reflectance-like and illumination-like representations, estimates a darkness prior, and uses learned attention and darkness-gated amplification to modulate illumination. It **does not** reproduce the eye's physical reflective layer, optical photon transport, or biological photoreceptor processes.

**Türkçe özet:** Görsel, Retinex'in ışık–yansıtım ve insan görsel algısı yaklaşımını biyolojik *tapetum lucidum* katmanının düşük ışıkta ışığı geri yansıtma işleviyle ilişkilendirir. Önerilen yöntem bu ilişkiyi fiziksel göz simülasyonu olarak değil, görüntü uzayında öğrenilebilir bir iyileştirme mekanizması olarak ele alır.

### 3. Retinex-Tapetum model architecture

<p align="center">
  <img src="TAPETUM/paper_retinextapetum/figures/tapetum_architecture_diagram.png" alt="Four-stage Retinex-Tapetum model architecture showing decomposition, auxiliary guidance, darkness-aware illumination enhancement, and bounded residual refinement" width="960">
</p>

This architecture implements the conceptual motivation using **Retinex Decomposition**, **Auxiliary Guidance Cues**, **Tapetum-Guided Illumination Enhancement**, and **Reconstruction and Bounded Residual Refinement**. The learned intermediate representations support diagnostic inspection; they are not independently verified causal explanations.

The figures provide a **conceptual research narrative**, not a complete physical camera simulator or proof of human-perceived image quality. See the [annotated figure gallery](./TAPETUM/paper_retinextapetum/figures/README.md) for interpretation details and original files.

## Research focus

- **Human-oriented visual quality:** improving low-light image legibility while assessing fidelity, structural quality, and perceptual differences.
- **Interpretable intermediate representations:** exposing reflectance-like and illumination-like components, darkness estimates, learned guidance and amplification maps, and bounded residual adjustments for inspection and diagnostic interpretation.
- **Auditable experimental protocols:** documenting the data splits, hyperparameter-search stages, candidate/seed/checkpoint selection, evaluation procedures, and speed-profiling conditions alongside publicly accessible Colab workflows.
- **Responsible interpretation:** treating attention and learned maps as diagnostic signals, not as proof of feature-level causal explanations.

## Evidence and audit trail

The Human-Centered AI research objective is supported by **inspectable intermediate representations** and **auditable experimental records** rather than an unsupported claim of explainability validation.

| Research dimension | Public research artifact | What can be examined |
|---|---|---|
| Model inspectability | [Model implementation](./TAPETUM/RetinexTapetum/model.py), [architecture diagram](./TAPETUM/paper_retinextapetum/figures/tapetum_architecture_diagram.png) | Reflectance-like and illumination-like estimates, darkness-derived cues, learned attention/amplification, bounded refinement |
| Diagnostic intervention | [Fixed-checkpoint intervention script](./TAPETUM/RetinexTapetum/RetinexTapetum_Ablation_Diagnostics_Dataset_Aware.py) | Changes in a trained network's processing behavior; **not** retrained architectural ablation evidence |
| Data split integrity | [LOL-v1 split manifest](./TAPETUM/HyperparameterSearch/lol_v1/split_manifest.json), [HPO records](./TAPETUM/HyperparameterSearch) | Fixed optimization/validation partitions and validation-only selection |
| Computational reproducibility | [Colab notebook](./TAPETUM/RetinexTapetum_ALL_models.ipynb), [reproduction instructions](./TAPETUM/README.md#quick-start) | Re-execution using the specified assets, configurations, and dependencies |
| Results and profiling | [Comparison artifacts](./TAPETUM/paper_retinextapetum/comparison_results), [speed logs](./TAPETUM/SpeedMetrics) | Evaluation outputs and recorded performance conditions |

**Evidence boundary:** Model-generated maps support inspection and diagnosis, but causal explanation faithfulness and benefits for human users have not been independently validated. The four paired benchmarks use **four dataset-specific selected checkpoints**, not one fixed universal model.

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
