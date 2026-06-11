Languages: [English](../../README.md) | [Deutsch](README.de.md) | [العربية](README.ar.md) | [Türkçe](README.tr.md) | [Français](README.fr.md)

# EEG-MRANC — Documentation open science multilingue

**Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** pour le débruitage d'EEG clinique en conditions réelles.

Développé par : **Ghadeer Mostafa** · [github.com/GhadeerMostafa/EEG-MRANC](https://github.com/GhadeerMostafa/EEG-MRANC)

---

## Introduction du projet

Le **Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** est un cadre de décomposition profonde à contrainte physique (*physics-locked*) pour l'EEG scalp à 32 canaux. Le modèle sépare chaque fenêtre en quatre composantes d'artefacts interprétables (EOG, EMG, ECG, bruit de base) plus une trace neuronale récupérée, tout en préservant une identité de reconstruction additive exacte (*physics-locked algebraic identity*).

L'entraînement suit la **Sequential Knowledge-Stacking Pipeline** : les poids sont transférés progressivement depuis le SSVEP Artifact Benchmark via DEAP et SEED, puis adaptés de manière paramétriquement efficace sur les fenêtres cliniques CHB-MIT. Un **Multi-Scale Attention Block (MSAB)** initialisé à zéro fournit des cartes de saillance temporelle pour une localisation explicable des artefacts sans déstabiliser les représentations empilées à l'initialisation.

Ce dépôt fournit le pipeline PyTorch complet pour l'entraînement, l'évaluation et la reproduction des benchmarks quantitatifs.

## Architecture (composants principaux)

- **Multi-Resolution Attention-Guided Neural Cleaner (MRANC)** — décomposition encodeur-décodeur avec quatre stems d'artefacts et reconstruction neuronale
- **Multi-Scale Attention Block (MSAB)** — attention temporelle multi-échelle pour la localisation explicable des artefacts
- **Squeeze-and-Excitation (SE) Gating** — pondération par canal pour stabiliser la représentation des caractéristiques
- **Sequential Knowledge-Stacking Pipeline** — transfert progressif des poids à travers les jeux de données de référence et d'émotion
- **Physics-locked algebraic identity** — reconstruction additive exacte : entrée = somme de tous les stems plus la trace neuronale

## Jeux de données validés

Les sources de données utilisées et validées dans ce dépôt sont :

- **SSVEP Artifact Benchmark** — pré-entraînement et référence d'artefacts
- **DEAP** — phase de transfert (EEG émotionnel)
- **SEED** — phase de transfert (EEG émotionnel)
- **CHB-MIT** (PhysioNet `chb01`) — évaluation clinique et adaptation paramétriquement efficace

## Limites et travaux futurs

L'évaluation clinique de ce dépôt se limite à **CHB-MIT (PhysioNet chb01)**. Le **Temple University Hospital (TUH) EEG Corpus** n'a ni été téléchargé ni évalué ici ; il est mentionné uniquement comme cible de validation clinique inter-institutionnelle future.

## Prérequis d'installation

- Python 3.10+ (recommandé : 3.12)
- GPU compatible **CUDA** (requis pour l'entraînement et l'évaluation par défaut)
- Git et `pip`
- Optionnel : LaTeX (`pdflatex`) pour la compilation du manuscrit

Cloner le dépôt et installer les dépendances :

```bash
git clone https://github.com/GhadeerMostafa/EEG-MRANC.git
cd EEG-MRANC
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
py src/setup_folders.py
py scripts/check_dependencies.py
```

*Remarque :* sous Linux/macOS, l'activation est `source .venv/bin/activate`. Sous Windows, exécutez les scripts depuis la racine du dépôt avec `py scripts/<nom>.py`.

PyTorch CUDA (Windows, roues hors ligne) :

```powershell
.\scripts\install_torch_local.ps1
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Exécution des scripts

Exécutez toutes les commandes depuis la **racine du dépôt**.

### Flux de travail principal

```text
1. Placer les données brutes  ->  data/raw/{deap,seed,clinical,artifact_benchmark}/
2. Convertir / prétraiter     ->  scripts spécifiques sous scripts/
3. Entraînement (stacking)    ->  py scripts/run_sequential_stacking.py
4. Métriques                  ->  py scripts/evaluate_metrics.py --dataset <nom>
5. Figures                    ->  py scripts/plot_results.py --dataset <nom>
6. Rapport                    ->  py scripts/run_report.py [--refresh | --full-refresh]
```

### Préparation des données

| Jeu de données | Dossier brut | Commande de conversion |
|----------------|--------------|------------------------|
| DEAP | `data/raw/deap/` | `py scripts/deap/convert_batch_2.py` |
| SEED | `data/raw/seed/` | `py scripts/seed/extract_and_convert_seed.py` |
| Clinique (CHB-MIT) | `data/raw/clinical/` | `py scripts/clinical/download_clinical.py` puis `preprocess_clinical.py` |
| SSVEP Artifact Benchmark | `data/raw/artifact_benchmark/` | `bash scripts/download_artifact_benchmark.sh` puis `py scripts/preprocess_artifact_benchmark.py` |

### Entraînement et évaluation

```bash
py scripts/run_sequential_stacking.py
py scripts/evaluate_metrics.py --dataset all
py scripts/evaluate_metrics.py --dataset clinical
py scripts/run_report.py --refresh
```

Phases de la Sequential Knowledge-Stacking Pipeline : pré-entraînement SSVEP Artifact Benchmark, transfert DEAP, transfert SEED, adaptateur d'attention clinique. Poids sous `artifacts/models/checkpoints/` ; chemins actuels dans `artifacts/models/checkpoints/stacking_latest.json`.

---

Documentation complète en anglais : [../../README.md](../../README.md)
