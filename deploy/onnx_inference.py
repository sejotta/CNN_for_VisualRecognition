"""Minimal ONNX Runtime inference for a PC, Raspberry Pi or other CPU device.

This script deliberately does not import PyTorch. Export the model once on the
training machine with ``python -m landmark_classifier.export_onnx`` and copy the
resulting ``.onnx`` file here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC


def collect_images(inputs: Sequence[Path]) -> list[Path]:
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
        raise ValueError("No supported image files were found.")
    return unique


def preprocess(
    image: Image.Image,
    image_size: tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Match the evaluation transform: resize shorter side, center crop, normalize."""
    height, width = image_size
    resize_size = int(round(max(height, width) * 256 / 224))

    image = image.convert("RGB")
    source_width, source_height = image.size
    if source_width <= source_height:
        new_width = resize_size
        new_height = int(round(source_height * resize_size / source_width))
    else:
        new_height = resize_size
        new_width = int(round(source_width * resize_size / source_height))

    image = image.resize((new_width, new_height), RESAMPLE)

    left = max(0, (new_width - width) // 2)
    top = max(0, (new_height - height) // 2)
    image = image.crop((left, top, left + width, top + height))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def load_metadata(session: ort.InferenceSession) -> dict:
    metadata = session.get_modelmeta().custom_metadata_map
    required = {"class_names", "image_size", "mean", "std"}
    missing = sorted(required - set(metadata))
    if missing:
        raise KeyError(
            "The ONNX file is missing preprocessing metadata: "
            f"{missing}. Re-export it with landmark_classifier.export_onnx."
        )
    return {
        "class_names": json.loads(metadata["class_names"]),
        "image_size": tuple(int(value) for value in json.loads(metadata["image_size"])),
        "mean": json.loads(metadata["mean"]),
        "std": json.loads(metadata["std"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run landmark classification with ONNX Runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        required=True,
        help="One or more image files or directories.",
    )
    parser.add_argument("--topk", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.topk < 1:
        raise ValueError("--topk must be >= 1.")

    session = ort.InferenceSession(
        str(args.model),
        providers=["CPUExecutionProvider"],
    )
    metadata = load_metadata(session)
    class_names = metadata["class_names"]
    image_size = metadata["image_size"]
    mean = metadata["mean"]
    std = metadata["std"]
    input_name = session.get_inputs()[0].name

    paths = collect_images(args.image)
    print(f"Model: {args.model}")
    print(f"Classes: {class_names}")
    print(f"Images: {len(paths)}")

    for path in paths:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        tensor = preprocess(image, image_size, mean, std)
        logits = session.run(None, {input_name: tensor})[0][0]
        probabilities = softmax(logits)
        order = np.argsort(probabilities)[::-1][: args.topk]

        print(f"\n{path}")
        print(
            "  top-k: "
            + " | ".join(
                f"{class_names[int(index)]} {float(probabilities[index]):.2%}"
                for index in order
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
