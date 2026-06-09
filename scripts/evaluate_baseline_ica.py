"""
Empirical Traditional ICA baseline on validation windows (clinical, deap, seed).
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

import numpy as np
import torch
from runtime import setup_src_path

setup_src_path()

try:
    from sklearn.decomposition import FastICA
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from exc

from baseline_eval import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_FRACTION,
    baseline_ica_metrics_path,
    cap_val_indices,
    dataset_length,
    ensure_reports_dir,
    list_baseline_datasets,
    load_mix_batch,
    load_refs_batch,
    metrics_dict_to_report,
    resolve_data_dir,
    resolve_ica_max_windows,
    resolve_sample_rate,
    save_baseline_json,
    val_indices,
)
from evaluate_metrics import aggregate_metrics, compute_batch_metrics_torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FRONTAL_CHANNEL_INDICES = (0, 1)
TEMPORAL_CHANNEL_INDICES = (7, 20)
RMS_RATIO_HIGH = 1.5
RMS_RATIO_LOW = 0.02
RMS_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FastICA baseline on validation splits")
    p.add_argument("--dataset", type=str, default="all", help="clinical, deap, seed, or all")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--corr-threshold", type=float, default=0.35)
    p.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap val windows (default: all for clinical/seed, 512 for deap)",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=16,
        help="FastICA components per window (lower = faster)",
    )
    p.add_argument("--skip-if-exists", action="store_true", help="Skip dataset if metrics JSON exists")
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    return p.parse_args()


def _zscore_1d(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float64).ravel()
    if a.size < 2:
        return a - a.mean()
    std = float(a.std())
    if std < 1e-12:
        return a - a.mean()
    return (a - a.mean()) / std


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    za = _zscore_1d(a)
    zb = _zscore_1d(b)
    if za.size < 2:
        return 0.0
    denom = np.linalg.norm(za) * np.linalg.norm(zb) + 1e-12
    return float(np.dot(za, zb) / denom)


def _ref_correlation(source: np.ndarray, refs: dict[str, np.ndarray]) -> float:
    z_source = _zscore_1d(source)
    best = 0.0
    for key in ("ref_eog", "ref_emg", "ref_ecg"):
        ref = refs[key]
        for ch in range(ref.shape[1]):
            z_ref = _zscore_1d(ref[0, ch])
            denom = np.linalg.norm(z_source) * np.linalg.norm(z_ref) + 1e-12
            best = max(best, abs(float(np.dot(z_source, z_ref) / denom)))
    return best


def _mixing_column(ica: FastICA, comp_idx: int) -> np.ndarray:
    if hasattr(ica, "mixing_") and ica.mixing_ is not None:
        return np.asarray(ica.mixing_[:, comp_idx]).ravel()
    return np.asarray(ica.components_[comp_idx]).ravel()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def _amplitude_guard(
    mix_ct: np.ndarray,
    clean_ct: np.ndarray,
    *,
    low: float = RMS_RATIO_LOW,
    high: float = RMS_RATIO_HIGH,
) -> tuple[np.ndarray, bool]:
    """Reject pathological scale (explosion or near-zero); allow normal artifact attenuation."""
    rms_mix = _rms(mix_ct)
    if rms_mix < RMS_EPS:
        return mix_ct.copy(), True
    rms_clean = _rms(clean_ct)
    ratio = rms_clean / rms_mix
    if ratio > high or ratio < low:
        return mix_ct.copy(), True
    return clean_ct, False


def _channel_load(mixing_col: np.ndarray, channel_indices: tuple[int, ...]) -> float:
    """Mean |loading| on selected channels (use list index — tuple index is 2D on 1D arrays)."""
    col = np.asarray(mixing_col).ravel()
    idx = [i for i in channel_indices if i < col.size]
    if not idx:
        return 0.0
    return float(np.mean(np.abs(col[idx])))


def _clinical_reject_component(
    comp_idx: int,
    source: np.ndarray,
    mixing_col: np.ndarray,
    sample_rate: float,
) -> bool:
    kurt = float(np.mean((source - source.mean()) ** 4) / (source.std() + 1e-8) ** 4)
    frontal_load = _channel_load(mixing_col, FRONTAL_CHANNEL_INDICES)
    temporal_load = _channel_load(mixing_col, TEMPORAL_CHANNEL_INDICES)
    if kurt > 5.0 and frontal_load > 0.15:
        return True
    if temporal_load > 0.2:
        spec = np.abs(np.fft.rfft(source))
        freqs = np.fft.rfftfreq(source.size, d=1.0 / sample_rate)
        high = spec[freqs > 30.0].sum()
        low = spec[freqs <= 30.0].sum() + 1e-8
        if high / low > 0.5:
            return True
    return False


def ica_denoise_window(
    mix_ct: np.ndarray,
    refs: dict[str, np.ndarray] | None,
    corr_threshold: float,
    *,
    n_components: int,
    sample_rate: float,
) -> tuple[np.ndarray, bool]:
    """mix_ct: (32, T) -> denoised (32, T)."""
    x = mix_ct.T.astype(np.float64)
    n_features = x.shape[1]
    n_comp = min(n_components, n_features, x.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ica_kw: dict = {
            "n_components": n_comp,
            "random_state": 42,
            "max_iter": 250,
            "tol": 5e-3,
            "whiten": "unit-variance",
        }
        try:
            ica = FastICA(**ica_kw, warn_on_nonconvergence=False)
        except TypeError:
            ica = FastICA(**ica_kw)
        try:
            sources = ica.fit_transform(x)
        except Exception as exc:
            logger.warning("FastICA failed for one window (%s); returning input", exc)
            return mix_ct.copy(), False

    reject: set[int] = set()
    for i in range(n_comp):
        if refs is not None:
            if _ref_correlation(sources[:, i], refs) >= corr_threshold:
                reject.add(i)
        elif _clinical_reject_component(
            i, sources[:, i], _mixing_column(ica, i), sample_rate
        ):
            reject.add(i)

    if not reject and refs is None:
        variances = np.var(sources, axis=0)
        top = int(np.argmax(variances))
        if _clinical_reject_component(
            top, sources[:, top], _mixing_column(ica, top), sample_rate
        ):
            reject.add(top)

    sources_clean = sources.copy()
    for idx in reject:
        sources_clean[:, idx] = 0.0

    x_clean = ica.inverse_transform(sources_clean)
    clean_ct = x_clean.T.astype(np.float32)
    return _amplitude_guard(mix_ct, clean_ct)


def evaluate_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> None:
    out_path = baseline_ica_metrics_path(dataset)
    if args.skip_if_exists and out_path.is_file():
        logger.info("Skipping %s (exists): %s", dataset, out_path)
        return

    data_dir = resolve_data_dir(dataset, args.data_dir)
    n_total = dataset_length(data_dir)
    val_idx_full = val_indices(n_total, args.val_fraction, args.split_seed)
    max_w = resolve_ica_max_windows(dataset, args.max_windows)
    val_idx, subsampled = cap_val_indices(val_idx_full, max_w, args.split_seed + 17)

    fs = resolve_sample_rate(dataset)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    logger.info(
        "ICA %s: evaluating %d / %d val windows (batch=%d, n_components=%d)%s",
        dataset,
        len(val_idx),
        len(val_idx_full),
        args.batch_size,
        args.n_components,
        " [subsampled]" if subsampled else "",
    )

    per_window_ica: list[dict] = []
    per_window_raw: list[dict] = []
    n_batches = (len(val_idx) + args.batch_size - 1) // args.batch_size
    amplitude_fallbacks = 0

    for batch_num, start in enumerate(range(0, len(val_idx), args.batch_size), start=1):
        batch_ids = val_idx[start : start + args.batch_size]
        mix_batch = load_mix_batch(data_dir, batch_ids)
        refs_batch = load_refs_batch(data_dir, batch_ids)

        clean_list = []
        for b, window_idx in enumerate(batch_ids):
            refs_one = None
            if refs_batch is not None:
                refs_one = {
                    "ref_eog": refs_batch["ref_eog"][b : b + 1],
                    "ref_emg": refs_batch["ref_emg"][b : b + 1],
                    "ref_ecg": refs_batch["ref_ecg"][b : b + 1],
                }
            clean_out, used_fallback = ica_denoise_window(
                mix_batch[b],
                refs_one,
                args.corr_threshold,
                n_components=args.n_components,
                sample_rate=fs,
            )
            if used_fallback:
                amplitude_fallbacks += 1
            clean_list.append(clean_out)
        clean_np = np.stack(clean_list, axis=0)

        if batch_num == 1 or batch_num % 5 == 0 or batch_num == n_batches:
            logger.info("ICA %s: batch %d / %d", dataset, batch_num, n_batches)

        mix_t = torch.from_numpy(mix_batch).to(device)
        clean_t = torch.from_numpy(clean_np).to(device)
        artifact_t = mix_t - clean_t
        zero_art = torch.zeros_like(mix_t)

        ica_metrics, _ = compute_batch_metrics_torch(mix_t, clean_t, artifact_t, fs)
        raw_metrics, _ = compute_batch_metrics_torch(mix_t, mix_t, zero_art, fs)

        for j in range(len(batch_ids)):
            per_window_ica.append({k: float(ica_metrics[k][j].cpu()) for k in ica_metrics})
            per_window_raw.append({k: float(raw_metrics[k][j].cpu()) for k in raw_metrics})

    aggregated_ica = aggregate_metrics(per_window_ica)
    aggregated_raw = aggregate_metrics(per_window_raw)

    ensure_reports_dir(dataset)
    payload = metrics_dict_to_report(
        aggregated_ica,
        dataset=dataset,
        method="ica",
        n_val=len(val_idx),
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        raw_metrics=aggregated_raw,
    )
    payload["n_val_windows_total"] = len(val_idx_full)
    payload["val_subsampled"] = subsampled
    if subsampled:
        payload["val_subsample_max"] = max_w
    payload["ica_n_components"] = args.n_components
    payload["amplitude_fallback_windows"] = amplitude_fallbacks
    save_baseline_json(out_path, payload)
    if amplitude_fallbacks:
        logger.warning(
            "ICA %s: %d / %d windows used identity fallback (RMS amplitude guard)",
            dataset,
            amplitude_fallbacks,
            len(val_idx),
        )
    logger.info("Wrote ICA baseline metrics to %s", out_path)


def main() -> None:
    args = parse_args()
    try:
        datasets = list_baseline_datasets(args.dataset)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    for ds in datasets:
        try:
            evaluate_dataset(ds, args)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error("ICA evaluation failed for %s: %s", ds, exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
