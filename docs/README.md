# Documentation index

Guides for the MRANC EEG artifact separation project ([EEG-MRANC](https://github.com/GhadeerMostafa/EEG-MRANC)).

**New clone?** Start with **[getting-started.md](getting-started.md)** — raw data placement, training stack, metrics, plots, and `run_report.py`.

| Document | Contents |
|----------|----------|
| [getting-started.md](getting-started.md) | GitHub clone → preprocess → train → metrics → figures → report |
| [setup.md](setup.md) | Python, CUDA, dependencies, folder layout |
| [commands.md](commands.md) | Every runnable script and CLI flags |
| [datasets.md](datasets.md) | DEAP, SEED, clinical, artifact-benchmark pipelines |
| [clinical.md](clinical.md) | CHB-MIT download, preprocessing, UDA |
| [training.md](training.md) | Losses, sequential stacking, checkpoints |
| [figures.md](figures.md) | Critical figures, baselines, Word/LaTeX report |

## License

The codebase is **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** — see [LICENSE](../LICENSE) in the repo root.

## What git does not ship

Raw EEG, processed `.npy`, weights, generated PNG/JSON reports, and compiled Word/LaTeX manuscripts are **local outputs**. The repo provides scripts, `docs/manuscript/` source text for `run_report.py`, and empty folders (`.gitkeep`) so you can drop in data and regenerate everything.

Cursor IDE metadata (`.cursor/`) is excluded via `.gitignore`.

Run all scripts from the **repository root** unless noted otherwise.
