"""Checkpoint serialization and reconstruction."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import LandmarkClassifier

REQUIRED_KEYS = (
    "model_name",
    "class_names",
    "image_size",
    "mean",
    "std",
    "state_dict",
)


def save_checkpoint(
    path: str | Path,
    model: LandmarkClassifier,
    metadata: dict,
) -> Path:
    """Save model weights plus everything needed for standalone inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(metadata)
    payload["format_version"] = 1
    payload["state_dict"] = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint and validate that inference metadata is present."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint {path} is missing required metadata: {missing}. "
            "Use a checkpoint produced by this repository's training script."
        )
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: dict,
    device: str | torch.device,
) -> LandmarkClassifier:
    """Rebuild the exact architecture and load the trained weights."""
    model = LandmarkClassifier(
        model_name=checkpoint["model_name"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
