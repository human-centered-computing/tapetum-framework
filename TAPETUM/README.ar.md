# TAPETUM

[English](README.md) · [Türkçe](README.tr.md) · [Deutsch](README.de.md) · [العربية](README.ar.md)

> هذه خلاصة مترجمة قيد المراجعة وليست ترجمة كاملة. يرجع إلى [المصدر الإنجليزي](README.md) للتفاصيل ومعايير القبول.

مساحة بحث لتحسين RGB منخفض الإضاءة مع Retinex-Tapetum وتقييم متعدد النماذج. الخرائط الشبيهة بالانعكاس والإضاءة وأولوية الظلام والانتباه والتضخيم والتصحيح المحدود تدعم الفحص لا محاكاة العين أو تفسيراً سببياً أو منفعة مثبتة. 536,463 معاملاً وأربع نقاط تحقق خاصة بالبيانات، نتائج المخطوطة لا إعادة مستقلة. HPO على التحقق؛ LOL-v1 وLOL-v2 Real/Synthetic وUHD-LL down4 وDICM/LIME/MEF/NPE/VV؛ لا تفوق شامل. يتوقع Colab البيانات والأوزان في /content/drive/MyDrive/TAPETUM؛ الاستنساخ وحده لا يكفي. Quick Start والبيانات والنقاط في الأصل؛ استخدم الإعداد المختار لكل مجموعة. مخطوطة 2026 لـ Murat Delen وSerdar Ciftci دون ادعاء قبول نشر. AGPL-3.0.

## روابط المصدر

- [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
- [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
- [![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00.svg)
- [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)
- [paper_retinextapetum](./paper_retinextapetum)
- [RetinexTapetum_ALL_models.ipynb](./RetinexTapetum_ALL_models.ipynb)
- [`RetinexTapetum/model.py`](./RetinexTapetum/model.py)
- [architecture figure](./paper_retinextapetum/figures/tapetum_architecture_diagram.png)
- [LOL-v1](./HyperparameterSearch/lol_v1/split_manifest.json)
- [LOL-v2 Real-Captured](./HyperparameterSearch/lol_v2_real/split_manifest.json)
- [LOL-v2 Synthetic](./HyperparameterSearch/lol_v2_synthetic/split_manifest.json)
- [UHD-LL down4](./HyperparameterSearch/uhd_ll_down4/split_manifest.json)
- [HPO records](./HyperparameterSearch)
- [Fixed-checkpoint intervention code](./RetinexTapetum/RetinexTapetum_Ablation_Diagnostics_Dataset_Aware.py)
- [Comparison outputs](./paper_retinextapetum/comparison_results)
- [speed records](./SpeedMetrics)
- [Colab notebook](./RetinexTapetum_ALL_models.ipynb)
- [Retinex perception → camera pipeline → RGB observation](./paper_retinextapetum/figures/from_real_to_enhancement_retinex_tapetum.png)
- [Retinex and biological tapetum lucidum](./paper_retinextapetum/figures/tapedum_lucidum.png)
- [Four-stage Retinex-Tapetum architecture](./paper_retinextapetum/figures/tapetum_architecture_diagram.png)
- [annotated research figure gallery](./paper_retinextapetum/figures/README.md)
- [HyperparameterSearch](./HyperparameterSearch)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
- [shared Drive archive](https://drive.google.com/drive/folders/13ayyEC3V1wWdX3AXdfL8y7VqnL8eTPFT?usp=sharing)
- [SpeedMetrics](./SpeedMetrics)
- [LICENSE](./LICENSE)