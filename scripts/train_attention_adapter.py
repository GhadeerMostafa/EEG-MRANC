"""
Parameter-efficient fine-tuning for MRANC MultiScaleAttentionBlock on clinical data.

Only ms_attention (temporal branches, fusion, SE, out_proj) is trainable.
Encoder and all four artifact heads remain frozen for every epoch.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from runtime import setup_src_path

setup_src_path()

from model import MRANC
from checkpoint_paths import resolve_stacking_latest_required
from paths import CHECKPOINT_PHASE3_SEED, FINAL_ATTENTION_WEIGHTS, PROCESSED_CLINICAL, WEIGHTS_DIR

from train import build_clinical_train_val_dataloaders, compute_losses_clinical

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ADAPTER_EPOCHS = 15
ADAPTER_LR = 1e-5
# Loosen feature distillation so the adapter can move off zero-init identity.
DISTILL_WEIGHT = 0.01
PHYSICS_WEIGHT = 1.0
ZERO_INIT_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MRANC multi-scale attention adapter")
    p.add_argument("--data-dir", type=str, default=str(PROCESSED_CLINICAL))
    p.add_argument(
        "--resume-weights",
        type=str,
        default=None,
        help="Base stacked weights (default: latest phase3 from stacking manifest)",
    )
    p.add_argument("--output", type=str, default=str(FINAL_ATTENTION_WEIGHTS))
    p.add_argument("--adapter-epochs", type=int, default=ADAPTER_EPOCHS)
    p.add_argument("--adapter-lr", type=float, default=ADAPTER_LR)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--feature-channels", type=int, default=128)
    p.add_argument("--storage", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--stabilize-epochs",
        type=int,
        default=0,
        help="Deprecated; ignored. Full-model unfreeze is disabled.",
    )
    p.add_argument(
        "--stabilize-lr",
        type=float,
        default=0.0,
        help="Deprecated; ignored.",
    )
    p.add_argument(
        "--distill-weight",
        type=float,
        default=DISTILL_WEIGHT,
        help="Weight on encoder feature MSE distillation (lower = more freedom to denoise)",
    )
    p.add_argument(
        "--physics-weight",
        type=float,
        default=PHYSICS_WEIGHT,
        help="Weight on clinical anti-collapse physics losses",
    )
    return p.parse_args()


def load_base_weights(model: MRANC, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Resume weights not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info("Checkpoint missing keys (expected for new adapter): %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])


def freeze_backbone(model: MRANC) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for head in (model.eog_head, model.emg_head, model.ecg_head, model.basenoise_head):
        for param in head.parameters():
            param.requires_grad = False
    for proj in (model.project_eog, model.project_emg, model.project_ecg):
        for param in proj.parameters():
            param.requires_grad = False


def set_adapter_only_trainable(model: MRANC) -> None:
    for param in model.parameters():
        param.requires_grad = False
    freeze_backbone(model)
    for param in model.ms_attention.parameters():
        param.requires_grad = True


def adapter_param_count(model: MRANC) -> int:
    return sum(p.numel() for p in model.ms_attention.parameters() if p.requires_grad)


def reset_out_proj_zero(model: MRANC) -> None:
    """Clinical PEFT starts from identity adapter: zero final projection."""
    nn.init.zeros_(model.ms_attention.out_proj.weight)
    nn.init.zeros_(model.ms_attention.out_proj.bias)


def verify_out_proj_zero_init(model: MRANC) -> None:
    w_max = float(model.ms_attention.out_proj.weight.abs().max().item())
    b_max = float(model.ms_attention.out_proj.bias.abs().max().item())
    if w_max >= ZERO_INIT_TOL or b_max >= ZERO_INIT_TOL:
        raise RuntimeError(
            f"out_proj must be zero-initialized before training (max weight={w_max}, max bias={b_max})"
        )
    logger.info("Verified out_proj zero initialization (identity mapping at step 0)")


def compute_batch_loss(
    model: MRANC,
    batch: dict[str, torch.Tensor],
    *,
    distill_weight: float,
    physics_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mix = batch["mix"]
    with torch.no_grad():
        frozen_feats = model.encoder(mix)

    attended_feats = model.ms_attention(frozen_feats)
    loss_mse = F.mse_loss(attended_feats, frozen_feats)

    out = model(mix)
    phys = compute_losses_clinical(out, mix=mix, dataset_mode="clinical")
    loss = distill_weight * loss_mse + physics_weight * phys["total_loss"]

    metrics = {
        "loss_mse": float(loss_mse.detach().item()),
        "loss_phys": float(phys["total_loss"].detach().item()),
        "loss_ortho": float(phys["loss_ortho"].detach().item()),
        "loss_tv": float(phys["loss_tv"].detach().item()),
        "loss_artifact_var": float(phys["loss_artifact_var"].detach().item()),
        "loss_stem_xcorr": float(phys["loss_stem_xcorr"].detach().item()),
        "loss_psd_dev": float(phys["loss_psd_dev"].detach().item()),
        "loss_art_push": float(phys["loss_art_push"].detach().item()),
        "total_loss": float(loss.detach().item()),
    }
    return loss, metrics


def run_epoch(
    model: MRANC,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    distill_weight: float,
    physics_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss_mse": 0.0,
        "loss_phys": 0.0,
        "loss_ortho": 0.0,
        "loss_tv": 0.0,
        "loss_artifact_var": 0.0,
        "loss_stem_xcorr": 0.0,
        "loss_psd_dev": 0.0,
        "loss_art_push": 0.0,
        "total_loss": 0.0,
    }
    n_batches = 0
    adapter_params = [p for p in model.ms_attention.parameters() if p.requires_grad]

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss, metrics = compute_batch_loss(
            model,
            batch,
            distill_weight=distill_weight,
            physics_weight=physics_weight,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if adapter_params:
                torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            optimizer.step()

        for key in totals:
            totals[key] += metrics[key]
        n_batches += 1

    if n_batches == 0:
        return totals
    return {k: v / n_batches for k, v in totals.items()}


def save_weights(model: MRANC, path: Path, args: argparse.Namespace) -> None:
    if path.is_file():
        raise FileExistsError(
            f"Refusing to overwrite existing weights at {path}. Use a new --output path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "feature_channels": args.feature_channels,
                "adapter_epochs": args.adapter_epochs,
                "adapter_lr": args.adapter_lr,
                "distill_weight": args.distill_weight,
                "physics_weight": args.physics_weight,
            },
        },
        path,
    )
    logger.info("Saved final weights to %s", path)


def main() -> None:
    args = parse_args()
    if args.stabilize_epochs > 0:
        logger.warning(
            "--stabilize-epochs is deprecated and ignored; only the attention adapter is trained."
        )

    if not torch.cuda.is_available():
        logger.error("CUDA is required for clinical adapter training.")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    mix_path = data_dir / "mix.npy"
    if not mix_path.is_file():
        logger.error(
            "Clinical data not found at %s. Run scripts/clinical/preprocess_clinical.py first.",
            mix_path,
        )
        sys.exit(1)

    device = torch.device("cuda")
    train_loader, val_loader, _ = build_clinical_train_val_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
        num_workers=args.num_workers,
        storage=args.storage,
        chunk_windows=2048,
    )

    model = MRANC(feature_channels=args.feature_channels).to(device)
    if args.resume_weights:
        resume_path = Path(args.resume_weights)
        if not resume_path.is_absolute():
            resume_path = Path(__file__).resolve().parent.parent / resume_path
    else:
        resume_path = resolve_stacking_latest_required("phase3_seed", CHECKPOINT_PHASE3_SEED)
    load_base_weights(model, resume_path, device)
    reset_out_proj_zero(model)
    verify_out_proj_zero_init(model)

    set_adapter_only_trainable(model)
    adapter_params = [p for p in model.ms_attention.parameters() if p.requires_grad]
    logger.info(
        "Training %d adapter parameters for %d epochs (lr=%g, distill=%g, physics=%g)",
        adapter_param_count(model),
        args.adapter_epochs,
        args.adapter_lr,
        args.distill_weight,
        args.physics_weight,
    )
    optimizer = torch.optim.Adam(adapter_params, lr=args.adapter_lr, eps=1e-8)

    for epoch in range(1, args.adapter_epochs + 1):
        tr = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            distill_weight=args.distill_weight,
            physics_weight=args.physics_weight,
        )
        va = run_epoch(
            model,
            val_loader,
            device,
            None,
            distill_weight=args.distill_weight,
            physics_weight=args.physics_weight,
        )
        logger.info(
            "Epoch %d/%d train_total=%.4f (mse=%.4f phys=%.4f art_var=%.4f xcorr=%.4f psd=%.4f) "
            "val_total=%.4f",
            epoch,
            args.adapter_epochs,
            tr["total_loss"],
            tr["loss_mse"],
            tr["loss_phys"],
            tr["loss_artifact_var"],
            tr["loss_stem_xcorr"],
            tr["loss_psd_dev"],
            va["total_loss"],
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path
    if out_path.is_file():
        raise FileExistsError(
            f"Output already exists at {out_path}. Pass a new --output path (stacking uses versioned names)."
        )
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    save_weights(model, out_path, args)


if __name__ == "__main__":
    main()
