Languages: [English](../../README.md) | [Deutsch](README.de.md) | [العربية](README.ar.md) | [Türkçe](README.tr.md) | [Français](README.fr.md)

# EEG-MRANC — Mehrsprachige Open-Science-Dokumentation

**Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** für die Entrauschung klinischer EEG-Daten in der Praxis.

Entwickelt von: **Ghadeer Mostafa** · [github.com/GhadeerMostafa/EEG-MRANC](https://github.com/GhadeerMostafa/EEG-MRANC)

---

## Projekteinführung

Der **Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** ist ein physikalisch gebundenes (*physics-locked*) Deep-Learning-Framework zur Zerlegung von 32-Kanal-Skalp-EEG. Das Modell trennt jedes Zeitfenster in vier interpretierbare Artefakt-Komponenten (EOG, EMG, ECG, Basisrauschen) sowie eine rekonstruierte neuronale Spur und wahrt dabei eine exakte additive Rekonstruktionsidentität (*physics-locked algebraic identity*).

Das Training folgt der **Sequential Knowledge-Stacking Pipeline**: Gewichte werden schrittweise vom SSVEP Artifact Benchmark über DEAP und SEED bis zur parameter-effizienten Anpassung auf klinische CHB-MIT-Fenster übertragen. Ein null-initialisierter **Multi-Scale Attention Block (MSAB)** liefert zeitliche Saliency-Karten zur erklärbaren Artefaktlokalisierung, ohne die gestapelten Repräsentationen bei der Initialisierung zu destabilisieren.

Dieses Repository enthält die vollständige PyTorch-Pipeline für Training, Evaluation und Reproduktion der quantitativen Benchmarks.

## Architektur (Kernkomponenten)

- **Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** — Encoder-Dekoder-Zerlegung mit vier Artefakt-Stems und neuronaler Rekonstruktion
- **Multi-Scale Attention Block (MSAB)** — mehrskalige zeitliche Aufmerksamkeit für erklärbare Artefaktlokalisierung
- **Squeeze-and-Excitation (SE) Gating** — kanalweise Gewichtung zur Stabilisierung der Merkmalsrepräsentation
- **Sequential Knowledge-Stacking Pipeline** — progressive Gewichtsübertragung über Benchmark- und Emotionsdatensätze hinweg
- **Physics-locked algebraic identity** — exakte additive Rekonstruktion: Eingang = Summe aller Stems plus neuronale Spur

## Validierte Datensätze

In diesem Repository werden folgende Datenquellen verwendet und validiert:

- **SSVEP Artifact Benchmark** — Vortraining und Artefakt-Benchmark
- **DEAP** — Transferlernphase (Emotions-EEG)
- **SEED** — Transferlernphase (Emotions-EEG)
- **CHB-MIT** (PhysioNet `chb01`) — klinische Evaluation und parameter-effiziente Anpassung

## Einschränkungen und zukünftige Arbeit

Die klinische Evaluation in diesem Repository beschränkt sich auf **CHB-MIT (PhysioNet chb01)**. Der **Temple University Hospital (TUH) EEG Corpus** wurde weder heruntergeladen noch evaluiert; er wird ausschließlich als Ziel für zukünftige institutionenübergreifende klinische Validierung genannt.

## Installationsvoraussetzungen

- Python 3.10+ (empfohlen: 3.12)
- **CUDA**-fähige GPU (erforderlich für Training und Standard-Evaluation)
- Git und `pip`
- Optional: LaTeX (`pdflatex`) für die Manuskript-Kompilierung

Repository klonen und Abhängigkeiten installieren:

```bash
git clone https://github.com/GhadeerMostafa/EEG-MRANC.git
cd EEG-MRANC
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
py src/setup_folders.py
py scripts/check_dependencies.py
```

*Hinweis:* Unter Linux/macOS lautet die Aktivierung `source .venv/bin/activate`. Unter Windows wird `py scripts/<name>.py` vom Repository-Stammverzeichnis aus ausgeführt.

CUDA-PyTorch (Windows, Offline-Wheels):

```powershell
.\scripts\install_torch_local.ps1
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Skripte ausführen

Alle Befehle vom **Repository-Stammverzeichnis** ausführen.

### Kernworkflow

```text
1. Rohdaten ablegen       ->  data/raw/{deap,seed,clinical,artifact_benchmark}/
2. Konvertieren           ->  dataset-spezifische Skripte unter scripts/
3. Training (Stacking)    ->  py scripts/run_sequential_stacking.py
4. Metriken               ->  py scripts/evaluate_metrics.py --dataset <name>
5. Abbildungen            ->  py scripts/plot_results.py --dataset <name>
6. Bericht                ->  py scripts/run_report.py [--refresh | --full-refresh]
```

### Datenvorbereitung

| Datensatz | Rohordner | Konvertierungsbefehl |
|-----------|-----------|----------------------|
| DEAP | `data/raw/deap/` | `py scripts/deap/convert_batch_2.py` |
| SEED | `data/raw/seed/` | `py scripts/seed/extract_and_convert_seed.py` |
| Klinisch (CHB-MIT) | `data/raw/clinical/` | `py scripts/clinical/download_clinical.py` dann `preprocess_clinical.py` |
| SSVEP Artifact Benchmark | `data/raw/artifact_benchmark/` | `bash scripts/download_artifact_benchmark.sh` dann `py scripts/preprocess_artifact_benchmark.py` |

### Training und Evaluation

```bash
py scripts/run_sequential_stacking.py
py scripts/evaluate_metrics.py --dataset all
py scripts/evaluate_metrics.py --dataset clinical
py scripts/run_report.py --refresh
```

Phasen der Sequential Knowledge-Stacking Pipeline: SSVEP Artifact Benchmark-Vortraining, DEAP-Transfer, SEED-Transfer, klinischer Attention-Adapter. Gewichte unter `artifacts/models/checkpoints/`; aktuelle Pfade in `artifacts/models/checkpoints/stacking_latest.json`.

---

Vollständige englische Dokumentation: [../../README.md](../../README.md)
