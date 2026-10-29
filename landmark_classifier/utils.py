"""Small shared helpers: seeding, devices, JSON and tensor utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make a run as reproducible as reasonably possible on CPU and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Benchmark mode speeds up fixed-shape convolutions. Full deterministic
    # mode is intentionally not forced because it is much slower on many GPUs.
    torch.backends.cudnn.benchmark = True


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` / ``mps`` into a torch device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return device


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)``."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: str | Path, payload: Any) -> None:
    """Write indented, UTF-8 JSON and create parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, ...] | list[float],
    std: tuple[float, ...] | list[float],
) -> np.ndarray:
    """Convert a normalized CHW tensor into an HWC float image in [0, 1]."""
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.asarray(std, dtype=np.float32)
    image = image + np.asarray(mean, dtype=np.float32)
    return np.clip(image, 0.0, 1.0)
