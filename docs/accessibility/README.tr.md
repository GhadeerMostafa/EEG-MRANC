Languages: [English](../../README.md) | [Deutsch](README.de.md) | [العربية](README.ar.md) | [Türkçe](README.tr.md) | [Français](README.fr.md)

# EEG-MRANC — Çok Dilli Açık Bilim Dokümantasyonu

Gerçek dünya klinik EEG gürültü giderme için **Multi-Resolution Attention-Guided Neural Cleaner (MRANC)**.

Geliştirici: **Ghadeer Mostafa** · [github.com/GhadeerMostafa/EEG-MRANC](https://github.com/GhadeerMostafa/EEG-MRANC)

---

## Proje Tanıtımı

**Multi-Resolution Attention-Guided Neural Cleaner (MRANC)**, 32 kanallı skalp EEG için fizik-kilitli (*physics-locked*) bir derin ayrıştırma çerçevesidir. Model, her pencereyi dört yorumlanabilir artefakt bileşenine (EOG, EMG, ECG, taban gürültüsü) ve kurtarılmış bir nöral ize ayırırken tam toplamsal bir yeniden yapılandırma kimliğini (*physics-locked algebraic identity*) korur.

Eğitim, **Sequential Knowledge-Stacking Pipeline** izler: ağırlıklar SSVEP Artifact Benchmark üzerinden DEAP ve SEED'e aktarılır, ardından CHB-MIT klinik pencerelerinde parametre-verimli uyarlamaya geçilir. Sıfır-başlatılmış **Multi-Scale Attention Block (MSAB)**, yığılmış temsilleri başlangıçta bozmadan açıklanabilir artefakt konumlandırması için zamansal önem haritaları sağlar.

Bu depo, nicel kıyaslamaların yeniden üretilmesi için gereken tam PyTorch eğitim ve değerlendirme hattını içerir.

## Mimari (Temel Bileşenler)

- **Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** — dört artefakt stem'i ve nöral yeniden yapılandırma ile kodlayıcı-çözücü ayrıştırması
- **Multi-Scale Attention Block (MSAB)** — açıklanabilir artefakt konumlandırması için çok ölçekli zamansal dikkat
- **Squeeze-and-Excitation (SE) Gating** — özellik temsilini stabilize etmek için kanal düzeyinde ağırlıklandırma
- **Sequential Knowledge-Stacking Pipeline** — kıyaslama ve duygu veri kümeleri boyunca aşamalı ağırlık aktarımı
- **Physics-locked algebraic identity** — tam toplamsal yeniden yapılandırma: girdi = tüm stem'lerin toplamı artı nöral iz

## Doğrulanmış Veri Kümeleri

Bu depoda kullanılan ve doğrulanan veri kaynakları:

- **SSVEP Artifact Benchmark** — ön eğitim ve artefakt kıyaslaması
- **DEAP** — transfer öğrenme aşaması (duygu EEG)
- **SEED** — transfer öğrenme aşaması (duygu EEG)
- **CHB-MIT** (PhysioNet `chb01`) — klinik değerlendirme ve parametre-verimli uyarlama

## Sınırlamalar ve Gelecek Çalışmalar

Bu depodaki klinik değerlendirme yalnızca **CHB-MIT (PhysioNet chb01)** ile sınırlıdır. **Temple University Hospital (TUH) EEG Corpus** indirilmemiş veya değerlendirilmemiştir; yalnızca gelecekteki kurumlar arası klinik doğrulama hedefi olarak belirtilmektedir.

## Model Ağırlıkları ve Değerlendirme

Depo, yapılandırma ve ağırlık manifestlerini `stacking_latest.json` aracılığıyla izler. Fiziksel `.pth` ikili dosyaları, belirli yayın öncesi ve lisanslama protokollerine uyum için yerel olarak tutulur.

Değerlendirme betiklerini çalıştırmak için ilgili `.pth` dosyalarının `stacking_latest.json` tarafından eşlenen yollarda bulunması gerekir:
- Phase 1: `artifacts/models/checkpoints/best_mranc_artifact_benchmark_weights_20260530_full.pth`
- Phase 2: `artifacts/models/checkpoints/best_mranc_phase2_deap_20260530_full.pth`
- Phase 3: `artifacts/models/checkpoints/best_mranc_phase3_seed_20260530_full.pth`
- Phase 4: `artifacts/models/weights/mranc_final_attention_20260530_full.pth`
- Baseline: `artifacts/models/checkpoints/baseline_eegdenoisenet_20260611_014917.pth`

**Hakemler için not:** Önceden eğitilmiş model kontrol noktaları, dergi değerlendirme süreci boyunca inceleme amacıyla talep üzerine tam olarak sunulabilir.

Ağırlıklar yerinde olduğunda ve işlenmiş veriler mevcut olduğunda, aşağıdaki **Betikleri Çalıştırma** bölümünde açıklandığı gibi `py scripts/evaluate_metrics.py --dataset <ad>` veya `py scripts/run_report.py --refresh` komutlarını çalıştırın.

## Kurulum Ön Koşulları

- Python 3.10+ (önerilen: 3.12)
- **CUDA** destekli GPU (eğitim ve varsayılan değerlendirme için gerekli)
- Git ve `pip`
- İsteğe bağlı: LaTeX (`pdflatex`) el yazısı derlemesi için

Depoyu klonlayın ve bağımlılıkları yükleyin:

```bash
git clone https://github.com/GhadeerMostafa/EEG-MRANC.git
cd EEG-MRANC
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
py src/setup_folders.py
py scripts/check_dependencies.py
```

*Not:* Linux/macOS'ta etkinleştirme `source .venv/bin/activate` şeklindedir. Windows'ta komutlar depo kök dizininden `py scripts/<ad>.py` ile çalıştırılır.

CUDA PyTorch (Windows, çevrimdışı tekerlekler):

```powershell
.\scripts\install_torch_local.ps1
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Betikleri Çalıştırma

Tüm komutları **depo kök dizininden** çalıştırın.

### Temel İş Akışı

```text
1. Ham veri yerleştir     ->  data/raw/{deap,seed,clinical,artifact_benchmark}/
2. Dönüştür / ön işle     ->  scripts/ altındaki veri kümesi betikleri
3. Eğitim (stacking)      ->  py scripts/run_sequential_stacking.py
4. Metrikler              ->  py scripts/evaluate_metrics.py --dataset <ad>
5. Grafikler              ->  py scripts/plot_results.py --dataset <ad>
6. Rapor                  ->  py scripts/run_report.py [--refresh | --full-refresh]
```

### Veri Hazırlığı

| Veri Kümesi | Ham Klasör | Dönüştürme Komutu |
|-------------|------------|-------------------|
| DEAP | `data/raw/deap/` | `py scripts/deap/convert_batch_2.py` |
| SEED | `data/raw/seed/` | `py scripts/seed/extract_and_convert_seed.py` |
| Klinik (CHB-MIT) | `data/raw/clinical/` | `py scripts/clinical/download_clinical.py` ardından `preprocess_clinical.py` |
| SSVEP Artifact Benchmark | `data/raw/artifact_benchmark/` | `bash scripts/download_artifact_benchmark.sh` ardından `py scripts/preprocess_artifact_benchmark.py` |

### Eğitim ve Değerlendirme

```bash
py scripts/run_sequential_stacking.py
py scripts/evaluate_metrics.py --dataset all
py scripts/evaluate_metrics.py --dataset clinical
py scripts/run_report.py --refresh
```

Sequential Knowledge-Stacking Pipeline aşamaları: SSVEP Artifact Benchmark ön eğitimi, DEAP transferi, SEED transferi, klinik dikkat adaptörü. Ağırlıklar `artifacts/models/checkpoints/` altında; güncel yollar `artifacts/models/checkpoints/stacking_latest.json` dosyasında kayıtlıdır.

---

Tam İngilizce dokümantasyon: [../../README.md](../../README.md)
