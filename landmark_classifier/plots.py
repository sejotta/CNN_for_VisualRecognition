"""Matplotlib figures used by the training, evaluation and prediction CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # CLIs save figures; never block on an interactive window.

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DISPLAY_NAMES = {
    # The dataset folder kept the Spanish spelling; show the English name.
    "Tokio_Tower": "Tokyo Tower",
}


def pretty_class_name(name: str) -> str:
    """Presentation-only label; the model still uses the dataset folder name."""
    return DISPLAY_NAMES.get(name, name.replace("_", " "))


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    """Save a labelled confusion matrix and return the written path."""
    matrix = np.asarray(matrix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(name) for name in class_names]

    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    axis.set_title(title, fontsize=13)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_training_curves(
    history: Sequence[dict],
    output_path: str | Path,
) -> Path:
    """Save a two-panel loss/accuracy figure for the two training phases."""
    if not history:
        raise ValueError("history is empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history) + 1)
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    train_accuracy = [record["train_accuracy"] for record in history]
    val_accuracy = [record["val_accuracy"] for record in history]

    head_epochs = sum(1 for record in history if record.get("phase") == "head")
    phase_change = head_epochs + 0.5 if 0 < head_epochs < len(history) else None

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_loss, color="#d62728", linewidth=2.2, label="Training")
    axes[0].plot(epochs, val_loss, color="#1f77b4", linewidth=2.2, label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, color="#d62728", linewidth=2.2, label="Training")
    axes[1].plot(epochs, val_accuracy, color="#1f77b4", linewidth=2.2, label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if phase_change is not None:
        for axis in axes:
            axis.axvline(
                phase_change,
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                label="phase change",
            )
        axes[0].legend()

    figure.suptitle("Training history", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_prediction_grid(
    images: Sequence[Image.Image | np.ndarray],
    titles: Sequence[str],
    output_path: str | Path,
    colors: Sequence[str] | None = None,
    columns: int = 4,
) -> Path:
    """Save a small grid of predictions, avoiding huge figures for big folders."""
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length.")
    if not images:
        raise ValueError("images is empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limit = min(len(images), 12)
    columns = max(1, min(columns, limit))
    rows = int(np.ceil(limit / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.7 * rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for index in range(limit):
        image = images[index]
        if isinstance(image, Image.Image):
            preview = image.copy()
            preview.thumbnail((512, 512))
            array = np.asarray(preview.convert("RGB"))
        else:
            array = np.asarray(image)

        axis = flat_axes[index]
        axis.imshow(array)
        title_color = (
            colors[index]
            if colors is not None and index < len(colors)
            else "black"
        )
        axis.set_title(
            titles[index],
            fontsize=10,
            color=title_color,
        )
        axis.axis("off")

    for axis in flat_axes[limit:]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
