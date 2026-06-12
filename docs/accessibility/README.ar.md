Languages: [English](../../README.md) | [Deutsch](README.de.md) | [العربية](README.ar.md) | [Türkçe](README.tr.md) | [Français](README.fr.md)

<div dir="rtl" lang="ar">

# EEG-MRANC — وثائق العلوم المفتوحة متعددة اللغات

**Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** لتنقية تخطيط الدماغ الكهربائي السريري في البيئات الواقعية.

المطوّر: **Ghadeer Mostafa** · [github.com/GhadeerMostafa/EEG-MRANC](https://github.com/GhadeerMostafa/EEG-MRANC)

---

## مقدمة المشروع

يُعد **Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** إطارًا للتحليل العميق مقيدًا فيزيائيًا (*physics-locked*) لتخطيط الدماغ الكهربائي فروة الرأس بـ 32 قناة. يفصل النموذج كل نافذة زمنية إلى أربعة مكونات شوائب قابلة للتفسير (EOG، EMG، ECG، ضوضاء خط الأساس) بالإضافة إلى أثر عصبي مُستعاد، مع الحفاظ على هوية جبرية تجميعية دقيقة للإعادة البنائية (*physics-locked algebraic identity*).

يتبع التدريب **Sequential Knowledge-Stacking Pipeline**: تُنقل الأوزان تدريجيًا من SSVEP Artifact Benchmark عبر DEAP وSEED، ثم تُكيَّف بكفاءة معاملات على نوافذ CHB-MIT السريرية. يوفّر **Multi-Scale Attention Block (MSAB)** المُهيَّأ بقيم صفرية خرائط بروز زمنية لتحديد مواقع الشوائب بشكل قابل للتفسير دون إزعاج التمثيلات المكدّسة عند التهيئة.

يضم هذا المستودع خط أنابيب PyTorch الكامل للتدريب والتقييم وإعادة إنتاج المعايير الكمية.

## البنية المعمارية (المكوّنات الأساسية)

- **Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** — تحليل مُشفّر-مُفكّك بأربعة فروع للشوائب وإعادة بناء عصبية
- **Multi-Scale Attention Block (MSAB)** — انتباه زمني متعدد المقاييس لتحديد مواقع الشوائب بشكل قابل للتفسير
- **Squeeze-and-Excitation (SE) Gating** — ترجيح على مستوى القنوات لاستقرار تمثيل السمات
- **Sequential Knowledge-Stacking Pipeline** — نقل تدريجي للأوزان عبر مجموعات بيانات المعايير والعاطفة
- **Physics-locked algebraic identity** — إعادة بنائية تجميعية دقيقة: المدخل = مجموع جميع الفروع زائد الأثر العصبي

## مجموعات البيانات المُحقَّقة

مصادر البيانات المستخدمة والمُحقَّقة في هذا المستودع:

- **SSVEP Artifact Benchmark** — التدريب المسبق ومعيار الشوائب
- **DEAP** — مرحلة النقل (EEG عاطفي)
- **SEED** — مرحلة النقل (EEG عاطفي)
- **CHB-MIT** (PhysioNet `chb01`) — التقييم السريري والتكييف بكفاءة المعاملات

## القيود والعمل المستقبلي

يقتصر التقييم السريري في هذا المستودع على **CHB-MIT (PhysioNet chb01)** فقط.

## أوزان النموذج والتقييم

يتتبع المستودع التهيئة وقوائم أوزان النماذج عبر `stacking_latest.json`. تُحفظ ملفات `.pth` الثنائية محليًا للامتثال لبروتوكولات ما قبل النشر والترخيص المحددة.

لتشغيل برامج التقييم النصية، يجب أن تقع ملفات `.pth` المقابلة في المسارات المُعرَّفة في `stacking_latest.json`:
- Phase 1: `artifacts/models/checkpoints/best_mranc_artifact_benchmark_weights_20260530_full.pth`
- Phase 2: `artifacts/models/checkpoints/best_mranc_phase2_deap_20260530_full.pth`
- Phase 3: `artifacts/models/checkpoints/best_mranc_phase3_seed_20260530_full.pth`
- Phase 4: `artifacts/models/weights/mranc_final_attention_20260530_full.pth`
- Baseline: `artifacts/models/checkpoints/baseline_eegdenoisenet_20260611_014917.pth`

**ملاحظة للمحكمين:** نقاط تفتيش النموذج المُدرَّبة مسبقًا متاحة بالكامل لأغراض المراجعة عند الطلب خلال مرحلة تقييم المجلة.

بمجرد توفر الأوزان والبيانات المُعالَجة، نفّذ `py scripts/evaluate_metrics.py --dataset <اسم>` أو `py scripts/run_report.py --refresh` كما هو موضح في قسم **تشغيل البرامج النصية** أدناه.

## متطلبات التثبيت

- Python 3.10+ (موصى به: 3.12)
- وحدة معالجة رسوميات تدعم **CUDA** (مطلوبة للتدريب والتقييم الافتراضي)
- Git و`pip`
- اختياري: LaTeX (`pdflatex`) لتجميع المخطوطة

استنساخ المستودع وتثبيت التبعيات:

```bash
git clone https://github.com/GhadeerMostafa/EEG-MRANC.git
cd EEG-MRANC
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
py src/setup_folders.py
py scripts/check_dependencies.py
```

*ملاحظة:* على Linux/macOS يكون التفعيل `source .venv/bin/activate`. على Windows تُنفَّذ الأوامر من جذر المستودع بصيغة `py scripts/<اسم>.py`.

PyTorch مع CUDA (Windows، عجلات دون اتصال):

```powershell
.\scripts\install_torch_local.ps1
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## تشغيل البرامج النصية

نفّذ جميع الأوامر من **جذر المستودع**.

### سير العمل الأساسي

```text
1. وضع البيانات الخام     ->  data/raw/{deap,seed,clinical,artifact_benchmark}/
2. التحويل / المعالجة       ->  برامج نصية خاصة بكل مجموعة تحت scripts/
3. التدريب (stacking)         ->  py scripts/run_sequential_stacking.py
4. المقاييس                   ->  py scripts/evaluate_metrics.py --dataset <اسم>
5. الأشكال                    ->  py scripts/plot_results.py --dataset <اسم>
6. التقرير                    ->  py scripts/run_report.py [--refresh | --full-refresh]
```

### إعداد البيانات

| مجموعة البيانات | المجلد الخام | أمر التحويل |
|-----------------|--------------|-------------|
| DEAP | `data/raw/deap/` | `py scripts/deap/convert_batch_2.py` |
| SEED | `data/raw/seed/` | `py scripts/seed/extract_and_convert_seed.py` |
| سريري (CHB-MIT) | `data/raw/clinical/` | `py scripts/clinical/download_clinical.py` ثم `preprocess_clinical.py` |
| SSVEP Artifact Benchmark | `data/raw/artifact_benchmark/` | `bash scripts/download_artifact_benchmark.sh` ثم `py scripts/preprocess_artifact_benchmark.py` |

### التدريب والتقييم

```bash
py scripts/run_sequential_stacking.py
py scripts/evaluate_metrics.py --dataset all
py scripts/evaluate_metrics.py --dataset clinical
py scripts/run_report.py --refresh
```

مراحل Sequential Knowledge-Stacking Pipeline: التدريب المسبق على SSVEP Artifact Benchmark، النقل من DEAP، النقل من SEED، محوّل الانتباه السريري. تُحفظ الأوزان تحت `artifacts/models/checkpoints/`؛ المسارات الحالية في `artifacts/models/checkpoints/stacking_latest.json`.

---

الوثائق الكاملة بالإنجليزية: [../../README.md](../../README.md)

</div>
