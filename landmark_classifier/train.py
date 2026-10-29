"""Two-phase transfer learning for landmark classification.

Phase 1 trains only a new linear head while the pretrained backbone is frozen.
Phase 2 restores the best phase-1 weights, unfreezes the last N backbone stages
and fine-tunes them with a smaller learning rate.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import timm
import torch
from torch import nn

from .checkpoint import save_checkpoint
from .data import build_data_loaders
from .engine import evaluate, fit
from .model import LandmarkClassifier, freeze_backbone, unfreeze_last_blocks
from .plots import plot_training_curves
from .reporting import save_evaluation_artifacts
from .utils import count_parameters, resolve_device, save_json, set_seed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be strictly between 0 and 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a landmark classifier with two-phase transfer learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/mobilenetv3_large100"),
    )
    parser.add_argument(
        "--model",
        default="mobilenetv3_large_100.ra_in1k",
        help="Any timm model name. Tested: mobilenetv3_large_100.ra_in1k, "
        "tf_efficientnet_lite0.in1k.",
    )
    parser.add_argument("--epochs-head", type=_positive_int, default=8)
    parser.add_argument("--epochs-finetune", type=_positive_int, default=5)
    parser.add_argument("--lr-head", type=_positive_float, default=1e-3)
    parser.add_argument("--lr-finetune", type=_positive_float, default=1e-4)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--val-fraction", type=_fraction, default=0.25)
    parser.add_argument("--unfreeze-blocks", type=_non_negative_int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--patience",
        type=_non_negative_int,
        default=0,
        help="Early-stopping patience per phase; 0 disables early stopping.",
    )
    parser.add_argument("--seed", type=_non_negative_int, default=42)
    parser.add_argument("--num-workers", type=_non_negative_int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet weights (useful for offline smoke tests).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1).")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be >= 0.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    pretrained = not args.no_pretrained

    print(f"Device: {device}")
    print(f"Model:  {args.model} (pretrained={pretrained})")

    # Read the exact preprocessing contract from the backbone. This avoids
    # hard-coded 224/ImageNet values when a different timm model is selected.
    config_backbone = timm.create_model(args.model, pretrained=pretrained, num_classes=0)
    data_config = timm.data.resolve_model_data_config(config_backbone)
    del config_backbone
    gc.collect()

    image_size = tuple(int(value) for value in data_config["input_size"][1:])
    mean = tuple(float(value) for value in data_config["mean"])
    std = tuple(float(value) for value in data_config["std"])
    print(f"Input:  {image_size} | mean={mean} | std={std}")

    loaders = build_data_loaders(
        data_dir=args.data_dir,
        image_size=image_size,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"\nClasses ({len(loaders.class_names)}): {loaders.class_names}")
    print(
        "Train images per class: "
        + ", ".join(f"{name}={count}" for name, count in loaders.class_counts.items())
    )
    print(
        f"Split: {loaders.train_size} train | {loaders.val_size} validation | "
        f"{loaders.test_size} test"
    )

    model = LandmarkClassifier(
        model_name=args.model,
        num_classes=len(loaders.class_names),
        pretrained=pretrained,
        dropout=args.dropout,
    )
    total_parameters, _ = count_parameters(model)
    print(f"Total parameters: {total_parameters:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ------------------------------------------------------------------
    # Phase 1: train the new head with the pretrained backbone frozen.
    # ------------------------------------------------------------------
    freeze_backbone(model)
    model.to(device)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"\n=== Phase 1: head only ({args.epochs_head} epochs) ===\n"
        f"Trainable parameters: {trainable:,} / {total_parameters:,}"
    )

    optimizer_head = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )
    history_head, val_loss_head, state_head, epoch_head = fit(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        criterion=criterion,
        optimizer=optimizer_head,
        device=device,
        epochs=args.epochs_head,
        phase="head",
        patience=args.patience,
    )

    # ------------------------------------------------------------------
    # Phase 2: start from the best head weights and fine-tune the last
    # backbone stages with a smaller learning rate.
    # ------------------------------------------------------------------
    model.load_state_dict(state_head)
    unfrozen = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.to(device)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"\n=== Phase 2: partial fine-tuning ({args.epochs_finetune} epochs) ===\n"
        f"Unfrozen backbone stages: {unfrozen} | "
        f"Trainable parameters: {trainable:,} / {total_parameters:,}"
    )
    if args.unfreeze_blocks > 0 and unfrozen == 0:
        print(
            "Note: this backbone has no `blocks` container; "
            "the whole backbone was unfrozen instead."
        )

    optimizer_finetune = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr_finetune,
        weight_decay=args.weight_decay,
    )
    history_finetune, val_loss_finetune, state_finetune, epoch_finetune = fit(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        criterion=criterion,
        optimizer=optimizer_finetune,
        device=device,
        epochs=args.epochs_finetune,
        phase="finetune",
        patience=args.patience,
    )

    # Keep the best checkpoint across both phases, not merely the last epoch.
    if val_loss_finetune <= val_loss_head:
        best_state = state_finetune
        best_phase = "finetune"
        best_val_loss = val_loss_finetune
        best_epoch = epoch_finetune
    else:
        best_state = state_head
        best_phase = "head"
        best_val_loss = val_loss_head
        best_epoch = epoch_head

    model.load_state_dict(best_state)
    model.to(device)
    history = history_head + history_finetune

    # ------------------------------------------------------------------
    # Final evaluation on the untouched test set.
    # ------------------------------------------------------------------
    test_result = evaluate(
        model=model,
        loader=loaders.test,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        class_names=loaders.class_names,
    )

    print("\n=== Test set ===")
    print(f"Test loss:     {test_result.loss:.4f}")
    print(f"Test accuracy: {test_result.accuracy:.4f}")
    print(f"Macro F1:      {test_result.macro_f1:.4f}")

    # ------------------------------------------------------------------
    # Persist everything needed to inspect, reproduce or deploy the run.
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "history.json", history)

    test_metrics = save_evaluation_artifacts(
        output_dir=output_dir,
        result=test_result,
        class_names=loaders.class_names,
        paths=loaders.test_paths,
        prefix="test",
    )
    plot_training_curves(history, output_dir / "training_curves.png")

    save_checkpoint(
        output_dir / "best.pt",
        model,
        {
            "model_name": args.model,
            "class_names": loaders.class_names,
            "image_size": list(image_size),
            "mean": list(mean),
            "std": list(std),
            "dropout": args.dropout,
            "phase": best_phase,
            "epoch": best_epoch,
            "val_loss": float(best_val_loss),
            "test_metrics": test_metrics,
            "config": config,
        },
    )

    print(f"\nBest phase: {best_phase} (epoch {best_epoch})")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Artifacts written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
