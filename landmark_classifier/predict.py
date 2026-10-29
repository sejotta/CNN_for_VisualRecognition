"""Run a trained checkpoint on one image or a folder of images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps

from .checkpoint import build_model_from_checkpoint, load_checkpoint
from .data import build_eval_transform
from .plots import plot_prediction_grid
from .utils import resolve_device, save_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(inputs: Sequence[Path]) -> list[Path]:
    """Expand files/directories into a de-duplicated list of image paths."""
    collected: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            collected.extend(
                sorted(
                    path
                    for path in input_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
        elif input_path.is_file():
            collected.append(input_path)
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in collected:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    if not unique:
        raise ValueError("No supported image files were found in the given input.")
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict landmarks with a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more image files or directories.",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument(
        "--labels-from-folders",
        action="store_true",
        help="Infer the true label from the parent folder name (useful for test sets).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a JSON prediction file.",
    )
    parser.add_argument(
        "--save-grid",
        type=Path,
        default=None,
        help="Optional path for a PNG grid with up to 12 predictions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.topk < 1:
        raise ValueError("--topk must be >= 1.")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint, device)

    class_names = list(checkpoint["class_names"])
    transform = build_eval_transform(
        checkpoint["image_size"],
        checkpoint["mean"],
        checkpoint["std"],
    )

    paths = collect_images(args.input)
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Images: {len(paths)}")

    records: list[dict] = []
    grid_images: list[Image.Image] = []
    grid_titles: list[str] = []
    grid_colors: list[str] = []

    for path in paths:
        try:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - keep going for a whole folder
            print(f"\nSkipping {path}: {exc}")
            continue

        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        top = torch.topk(probabilities, k=min(args.topk, len(class_names)))
        predicted_index = int(top.indices[0])
        predicted_name = class_names[predicted_index]
        confidence = float(top.values[0])

        true_label = None
        if args.labels_from_folders:
            candidate = path.parent.name
            if candidate in class_names:
                true_label = candidate

        print(f"\n{path}")
        print(f"  prediction: {predicted_name} ({confidence:.2%})")
        print(
            "  top-k: "
            + " | ".join(
                f"{class_names[int(index)]} {float(value):.2%}"
                for value, index in zip(top.values, top.indices)
            )
        )
        if true_label is not None:
            marker = "OK" if true_label == predicted_name else "MISMATCH"
            print(f"  true label: {true_label} [{marker}]")

        records.append(
            {
                "path": str(path),
                "prediction": predicted_name,
                "confidence": confidence,
                "top_k": [
                    {
                        "class": class_names[int(index)],
                        "probability": float(value),
                    }
                    for value, index in zip(top.values, top.indices)
                ],
                "true_label": true_label,
            }
        )

        if true_label is None:
            title = f"{predicted_name} ({confidence:.1%})"
            color = "black"
        else:
            title = f"pred: {predicted_name} ({confidence:.1%})\ntrue: {true_label}"
            color = "green" if true_label == predicted_name else "red"
        grid_images.append(image)
        grid_titles.append(title)
        grid_colors.append(color)

    if not records:
        print("\nNo images could be read.")
        return 1

    if args.output_json is not None:
        save_json(
            args.output_json,
            {
                "checkpoint": str(args.checkpoint),
                "classes": class_names,
                "predictions": records,
            },
        )
        print(f"\nJSON written to: {args.output_json.resolve()}")

    if args.save_grid is not None:
        plot_prediction_grid(
            grid_images,
            grid_titles,
            args.save_grid,
            colors=grid_colors,
        )
        print(f"Grid written to: {args.save_grid.resolve()}")
        if len(grid_images) > 12:
            print("Grid shows the first 12 images.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
