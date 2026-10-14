# Retinex-Tapetum — Visual Research Gallery

These three existing figures connect **Retinex-based human scene perception**, **camera image acquisition and RGB observation**, **biological tapetum lucidum**, and the proposed **Retinex-Tapetum** computational pipeline. They support the research aim of **Human-Centered AI through interpretable intermediate representations and auditable experimental protocols**.

[← TAPETUM README](../../README.md) · [Repository homepage](../../../README.md) · [Open Colab notebook](https://colab.research.google.com/github/human-centered-computing/tapetum-framework/blob/main/TAPETUM/RetinexTapetum_ALL_models.ipynb)

## 1. Retinex perception, camera processing, and physical-scene information in RGB

![Conceptual diagram connecting human scene perception under Retinex, camera acquisition and image-signal processing, and resulting RGB observations](./from_real_to_enhancement_retinex_tapetum.png)

**Source file:** [`from_real_to_enhancement_retinex_tapetum.png`](./from_real_to_enhancement_retinex_tapetum.png)

**Read from left to right:**

| Part of the composite | Intended scientific meaning |
|---|---|
| **First / left — Retinex and perception** | Introduces theories of how humans perceive lightness and color in real scenes. Perception depends on illumination, surface reflectance and spatial/contextual relationships, rather than on raw sensor RGB values alone. The classical simplified Retinex model separates illumination and reflectance conceptually. |
| **Middle — camera image formation and processing** | Represents the imaging chain between the physical scene and stored digital image: optical acquisition, exposure/sensing and camera image processing (such as color processing and other ISP operations, depending on the system). |
| **Last / right — recorded RGB observation** | Represents the image available to computer-vision methods. Its RGB values reflect scene appearance **together with** effects of the camera pipeline, including possible sensor noise, ISP modifications and acquisition artifacts. It is neither a noise-free copy of the physical scene nor a uniquely separated reflectance/illumination representation. |

**Türkçe:** İlk bölüm gerçek sahnedeki renk ve aydınlık algısını açıklamakta kullanılan Retinex yaklaşımlarını; orta bölüm fotoğraf makinesinin görüntü oluşturma ve işleme adımlarını; son bölüm ise fiziksel sahne bilgisinin kameradaki işleme ve gürültü gibi etkilerle birleşerek kaydedilen RGB görüntüye dönüşmesini anlatır.

### Why this figure matters to Retinex-Tapetum

The framework operates on the **observed RGB image**, not on an unobservable ideal physical-scene image. Retinex decomposition estimates *learned reflectance-like and illumination-like representations* from the observed input. This factorization is modeling-dependent and should not be interpreted as a guaranteed physical recovery of scene reflectance or illumination. The diagram is a conceptual visual account, not a complete validated forward model of a particular camera's hardware or ISP.

## 2. Retinex and the biological tapetum lucidum

![Illustration linking Retinex-inspired visual perception with the biological tapetum lucidum low-light light-return mechanism](./tapedum_lucidum.png)

**Source file:** [`tapedum_lucidum.png`](./tapedum_lucidum.png)

This illustration links two **different layers of inspiration**:

1. **Retinex / perceptual foundation:** Illumination and reflectance provide a computational conceptualization of how a scene can appear under changing light, motivated by human color and lightness perception.
2. **Biological tapetum lucidum / low-light adaptation:** In many animals, a reflective layer behind the retina returns a portion of incident light through retinal photoreceptors, potentially improving light capture under dim illumination.
3. **Retinex-Tapetum / algorithmic abstraction:** The proposed method combines learned Retinex-like decomposition with a learned tapetum response `T` and spatial amplification map `Lambda`, gated by illumination darkness `D`. The illumination update is performed digitally: `L_t = L_low * (1 + Lambda * T)`.

**Türkçe:** Görselde Retinex'in aydınlatma–yansıtım ve görsel algı yaklaşımı, biyolojik *tapetum lucidum*'un düşük ışıkta ışığı fotoreseptörlere geri yönlendirme özelliği ve bu işlevden soyut düzeyde esinlenen Retinex-Tapetum yöntemi birlikte ele alınmaktadır.

**Essential distinction:** Retinex theory concerns scene perception/computational image modeling; tapetum lucidum is an optical biological structure. Retinex-Tapetum **does not** simulate photon transport, retinal physiology, or the physical reflective layer. The repository preserves the original filename `tapedum_lucidum.png` exactly.

## 3. How the conceptual ideas become a model

![Four-stage Retinex-Tapetum architecture with decomposition, guidance, illumination enhancement, and bounded refinement](./tapetum_architecture_diagram.png)

**Source file:** [`tapetum_architecture_diagram.png`](./tapetum_architecture_diagram.png)

The computation is organized into **Retinex Decomposition**, **Auxiliary Guidance Cues**, **Tapetum-Guided Illumination Enhancement**, and **Reconstruction and Bounded Residual Refinement**. Its observable model-generated intermediate quantities include `R_low`, `L_low`, `D`, `T`, `Lambda`, `Y_HF`, and bounded `Delta I`.

These representations support **mechanism-oriented inspection** and fixed-checkpoint diagnostic experiments. The maps are learned model signals; they are not direct physical measurements, formal causal attributions or independently validated explanations.

## Human-Centered AI: interpretation and evidence boundaries

This set of diagrams communicates a **research goal**: develop more inspectable intermediate representations and more auditable experimental protocols for computer vision. Source code, Colab workflows, fixed split manifests, hyperparameter-search records, test metrics and intervention scripts support this transparency objective.

The figures alone do **not** demonstrate formally validated explanation faithfulness, human-subject/user-centered evaluation, physiological equivalence, or superiority of a particular image for human viewers. Fixed-weight interventions also do not replace architectural ablations that retrain the networks.

## Reproduction and citation

See [TAPETUM reproducibility instructions](../../README.md#quick-start), [HPO records](../../HyperparameterSearch), and [manuscript details](../../README.md#paper). Respect applicable dataset and image-use licenses and cite the research artifacts appropriately.
