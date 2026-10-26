# Tapetum Framework

**Human-Centered AI · Interpretable Intermediate Representations · Auditable Experimental Protocols**

**Research objective:** To advance **Human-Centered AI** by developing **interpretable intermediate representations and auditable experimental protocols** for low-light image enhancement.

**Türkçe araştırma odağı:** İnsan merkezli yapay zekâ (Human-Centered AI) hedefi doğrultusunda, düşük ışık görüntü iyileştirme için yorumlanabilir ara temsiller ve denetlenebilir deney protokolleri geliştirmeye yönelik bir araştırma.

This repository documents **Retinex-Tapetum**, a bio-inspired, darkness-aware Retinex framework that investigates how explicit enhancement stages, inspectable guidance maps, and reproducible experimental records can support more transparent development and critical assessment of computer-vision systems. The human-centered framing is a **research goal and design orientation**, not a claim of validated human-user benefit or formally proven explanations.

The approach exposes intermediate representations such as reflectance-like and illumination-like maps, a darkness prior, a learned Tapetum Attention Map, and a Darkness-Gated Spatial Amplification Map. These offer *model inspectability* and support diagnostic interventions; they are **not** independently validated causal explanations or a substitute for user-centered studies.

## Research at a glance

**Retinex-Tapetum** is a compact, bio-inspired low-light image enhancement (LLIE) model operating on **already captured RGB images**. It uses learned Retinex-like decomposition, explicit darkness gating, bounded illumination amplification, and a bounded RGB residual. Normal-light reference images are required for **paired training**, not for inference.

| Manuscript-reported characteristic | Value / protocol |
|---|---|
| Trainable parameters | **536,463** (approximately 0.5365 M) |
| Computational complexity | **19.757 GMACs / 39.514 GFLOPs** at 256 × 256 |
| Mean GPU inference latency | **33.73 ms** |
| Mean GPU throughput | **29.72 FPS** |
| Runtime measurement | NVIDIA Tesla T4, FP32, batch size 1, 256 × 256, 50 warm-up and 500 timed iterations |
| Paired evaluation | LOL-v1, LOL-v2 Real-Captured, LOL-v2 Synthetic, UHD-LL down4 |
| Additional reference-free evaluation | DICM, LIME, MEF, NPE, VV |

**Reported headline result:** The four **separately trained, dataset-specific checkpoints** achieve the highest **SSIM** and lowest **LPIPS** among seven compared methods on all four paired benchmarks. They also achieve the highest **PSNR** on LOL-v2 Synthetic and UHD-LL down4—not on LOL-v1 or LOL-v2 Real-Captured. Full values and qualifications appear in [Benchmark results](#benchmark-results).

> The numbers above are taken from the accompanying research manuscript; this documentation update does **not** constitute an independent rerun. Runtime results are hardware-, software-, and protocol-dependent. The detailed [TAPETUM workspace README](./TAPETUM/README.md) also provides checkpoint-level speed-log summaries, whose arithmetic aggregates may differ slightly from the manuscript's reported summary.

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

## Method: explicit, bounded illumination control

The model has **four canonical stages**:

1. **Retinex Decomposition:** `DecomNet` estimates the **Reflectance-Like Representation** `R_low` and **Illumination-Like Representation** `L_low` from the low-light image `I_low`. This is a task-oriented factorization, **not** a physically unique decomposition.
2. **Auxiliary Guidance Cues:** an **Illumination-Derived Darkness Prior** `D` is computed from `L_low`; the **High-Frequency Luminance Cue** `Y_HF` is derived from `I_low`.
3. **Tapetum-Guided Illumination Enhancement:** the **Tapetum Attention Network** `F_T` predicts a three-channel **Tapetum Attention Map** `T` from `[I_low, L_low, D]`. The **Tapetum Amplification Network** `F_Λ` predicts a three-channel **Darkness-Gated Spatial Amplification Map** `Λ` from `[L_low, D]`.
4. **Reconstruction and Bounded Residual Refinement:** the updated illumination is combined with `R_low` into a **Retinex-Guided Base Reconstruction**; the **Residual Color–Detail Refinement Network** `F_ref` then applies a bounded correction.

The central update is:

$
L_t(x)=L_{\\mathrm{low}}(x)\\odot
\\left[1+\\Lambda(x)\\odot T(x)\\right],
\\qquad
I_{\\mathrm{base}}(x)=R_{\\mathrm{low}}(x)\\odot L_t(x).
$

The final output is:

$
\\Delta I(x)=0.08\\tanh\\left(F_{\\mathrm{ref}}(X_{\\mathrm{ref}})\\right),
\\qquad
I_{\\mathrm{enh}}(x)=\\operatorname{clip}\\left(I_{\\mathrm{base}}(x)+\\Delta I(x),0,1\\right).
$

Here `X_ref = [I_low, I_base, T, Λ, Y_HF]` contains **13 channels**. The multiplicative illumination gain is bounded between **1** and **1 + λ_max**, while each residual correction is bounded to **[−0.08, +0.08]**. These constraints limit the update magnitude; they do **not** guarantee restored image quality.

**Important routing and terminology details:** `Y_HF` is supplied **only** to final residual refinement—not to `F_T` or `F_Λ`. The learned `T` map performs spatial/chromatic **attention modulation**; it is **not** Transformer self-attention, multi-head attention, or query–key–value attention. In the source code, some legacy identifiers and filenames retain `RetinexTapetum`; **Retinex-Tapetum** is the canonical name in scientific documentation.

See the [implementation](./TAPETUM/RetinexTapetum/model.py) and [four-stage architecture figure](./TAPETUM/paper_retinextapetum/figures/tapetum_architecture_diagram.png).

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

## Benchmark results

The following **independent test-set** metrics are reported in the research manuscript for the **selected dataset-specific checkpoints**. PSNR and SSIM: higher is better; LPIPS: lower is better.

| Paired benchmark | Selected seed | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| LOL-v1 (15 test pairs) | 42 | 21.0392 | **0.8316** | **0.1357** |
| LOL-v2 Real-Captured (100) | 123 | 20.0190 | **0.8405** | **0.1258** |
| LOL-v2 Synthetic (100) | 3407 | **25.0530** | **0.9322** | **0.0577** |
| UHD-LL down4 (150) | 3407 | **24.7107** | **0.9115** | **0.0884** |

**Comparison context:** seven representative baselines—LIME, Zero-DCE, RUAS, RetinexNet, KinD++, URetinex-Net++, and RetinexFormer—were evaluated using a shared metric-calculation pipeline on stored outputs. RetinexFormer has the highest reported PSNR on LOL-v1 (21.6829 dB) and LOL-v2 Real-Captured (20.7071 dB). **KinD++'s compared inference configuration uses a normal-light-dependent adjustment ratio**, so it should not be described as strictly reference-free. URetinex-Net++ uses a train-calibrated blind protocol on the applicable paired evaluations.

### Evaluation design and selection integrity

| Dataset | Optimization pairs | Validation pairs | Independent test pairs |
|---|---:|---:|---:|
| LOL-v1 | 437 | 48 | 15 |
| LOL-v2 Real-Captured | 620 | 69 | 100 |
| LOL-v2 Synthetic | 810 | 90 | 100 |
| UHD-LL down4 | 1,884 | 116 | 150 |

Dataset-adaptive hyperparameter search has **four progressive stages** (optimizer, illumination, fidelity, and detail) with **10 / 10 / 12 / 14** target trials. The source-profile baseline and two distinct leading HPO configurations undergo independently initialized **three-seed confirmation** (`42`, `123`, `3407`). Configuration and representative-checkpoint decisions use validation metrics only, with a PSNR tolerance of **0.15 dB**, an SSIM tolerance of **0.002**, and LPIPS as the last selection criterion. Official test sets are excluded from these selection steps.

Relevant artifacts: [dataset split manifests and HPO records](./TAPETUM/HyperparameterSearch), [checkpoint metadata](./TAPETUM/RetinexTapetum/hyper_ckpt), [comparison outputs](./TAPETUM/paper_retinextapetum/comparison_results), and [speed-profiling records](./TAPETUM/SpeedMetrics).

### What the diagnostics show—and do not show

The reported experiments include **inference-time interventions on already trained checkpoints**, **not** separately retrained architecture ablations.

- Disabling the entire illumination update or spatial amplification strongly degrades results on all four paired benchmarks; **darkness-gated amplification is the principal functional enhancement pathway** in these checkpoints.
- Removing the learned Tapetum Attention Map's modulation has a smaller, sometimes nearly negligible, independent effect; the high-frequency cue likewise offers no consistently demonstrated benefit.
- Removing residual refinement **improves** the LOL-v2 Real-Captured checkpoint from **20.02 to 22.69 dB PSNR** (and also improves SSIM and LPIPS). Residual refinement is therefore **not uniformly beneficial** across selected checkpoints.

Cross-domain, no-reference evaluation on **DICM, LIME, MEF, NPE, and VV** produces mixed metric- and source-dependent rankings. These findings **do not** establish universal zero-shot superiority or prove the physical correctness or causal faithfulness of intermediate maps.

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

### Practical reproducibility notes

The [Colab notebook](./TAPETUM/RetinexTapetum_ALL_models.ipynb) is the main experiment entry point. A local clone provides code and compact records, but **not necessarily the datasets, baseline repositories, trained weights, or generated outputs** needed for complete reproduction. The [TAPETUM README](./TAPETUM/README.md#quick-start) documents the expected directory layout, environment selectors, and training/test commands.

**Do not assume that the default values in** [`config.py`](./TAPETUM/RetinexTapetum/config.py) **are the manuscript-selected HPO parameters.** Use the dataset-specific selection records and the corresponding selected checkpoint when trying to reproduce a reported benchmark result. Dataset distribution conditions and GPU/software differences can also affect repeatability.

## Scope and research limitations

Retinex-Tapetum is motivated by the functional light-reuse analogy of the *tapetum lucidum*, not a physically faithful simulation of the eye. The provided diagnostic maps are model-generated representations, and fixed-checkpoint interventions should not be interpreted as evidence from retrained architectural ablations. Formal explanation-faithfulness tests and studies with end users remain opportunities for future research.

## Publication, citation, and future direction

**Manuscript (2026):** Murat Delen and Serdar Ciftci, *Retinex-Tapetum: A Bio-Inspired Darkness-Aware Retinex Framework for Low-Light Image Enhancement*. The paper's results should be treated as **reported experimental findings**; no journal acceptance, publication venue, or DOI is asserted here.

Provisional BibTeX, until a formal publication record is available:

```bibtex
@misc{delen2026retinextapetum,
  title  = {Retinex-Tapetum: A Bio-Inspired Darkness-Aware Retinex Framework for Low-Light Image Enhancement},
  author = {Delen, Murat and Ciftci, Serdar},
  year   = {2026},
  note   = {Research manuscript and open-source framework},
  url    = {https://github.com/human-centered-computing/tapetum-framework}
}
```

**Planned—not yet validated in this repository:** offline enhancement of prerecorded drone/field video; temporal consistency and flicker assessment; and potential integration into **Tapetum Sentinel** for human-centered video analysis, including disaster-response imagery. These are future research directions, not demonstrated video or downstream detection results.

**Türkçe kısa özet:** Retinex-Tapetum, düşük ışıklı RGB görüntülerde Retinex ayrıştırmasını karanlıkla kapılanan, sınırlandırılmış aydınlatma güçlendirmesiyle birleştirir. Dört eşleştirilmiş test kümesinin tamamında karşılaştırılan yöntemler arasında en yüksek SSIM ve en düşük LPIPS değerleri raporlanmıştır. Bu sonuçlar veri kümesine özgü ayrı modellerden elde edilmiştir; insan kullanıcılarla doğrulanmış açıklanabilirlik ya da tüm veri kümelerine genellenebilir üstünlük iddiası değildir.

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](./LICENSE). Contributions are accepted under the same license.
