"""Evaluate a trained checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from .checkpoint import build_model_from_checkpoint, load_checkpoint
from .data import build_test_loader
from .engine import evaluate
from .reporting import classification_report_text, save_evaluation_artifacts
from .utils import resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a landmark-classifier checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write metrics/plots. If omitted, results are only printed.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint, device)

    class_names = list(checkpoint["class_names"])
    loader, dataset_classes, paths = build_test_loader(
        data_dir=args.data_dir,
        image_size=checkpoint["image_size"],
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        class_names=class_names,
    )

    print(f"Device: {device}")
    print(f"Classes: {dataset_classes}")
    print(f"Test images: {len(paths)}")

    result = evaluate(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        class_names=class_names,
    )

    print(f"\nTest loss:     {result.loss:.4f}")
    print(f"Test accuracy: {result.accuracy:.4f}")
    print(f"Macro F1:      {result.macro_f1:.4f}")
    print("\n" + classification_report_text(result, class_names))

    if args.output_dir is not None:
        metrics = save_evaluation_artifacts(
            output_dir=args.output_dir,
            result=result,
            class_names=class_names,
            paths=paths,
            prefix="test",
        )
        save_json(
            args.output_dir / "metrics.json",
            {
                "checkpoint": str(args.checkpoint),
                "images": len(paths),
                "test": metrics,
            },
        )
        print(f"Artifacts written to: {args.output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
