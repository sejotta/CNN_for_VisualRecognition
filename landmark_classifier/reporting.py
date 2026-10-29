"""Turn evaluation results into files that are useful outside the terminal."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report

from .engine import EvaluationResult
from .plots import plot_confusion_matrix, pretty_class_name
from .utils import save_json


def evaluation_to_dict(
    result: EvaluationResult,
    class_names: Sequence[str],
) -> dict:
    """Convert an :class:`EvaluationResult` into JSON-friendly metrics."""
    per_class = {}
    for name in class_names:
        row = result.report.get(name, {})
        per_class[name] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        }

    return {
        "class_names": list(class_names),
        "images": int(len(result.y_true)),
        "loss": float(result.loss),
        "accuracy": float(result.accuracy),
        "macro_f1": float(result.macro_f1),
        "weighted_f1": float(result.weighted_f1),
        "per_class": per_class,
        "confusion_matrix": result.confusion_matrix.astype(int).tolist(),
    }


def classification_report_text(
    result: EvaluationResult,
    class_names: Sequence[str],
) -> str:
    return classification_report(
        result.y_true,
        result.y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )


def write_predictions_csv(
    path: str | Path,
    paths: Sequence[str],
    result: EvaluationResult,
    class_names: Sequence[str],
) -> Path:
    """Write one row per test image with the true and predicted label."""
    if len(paths) != len(result.y_pred):
        raise ValueError(
            "The number of image paths does not match the number of predictions."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    confidences = result.probabilities[np.arange(len(result.y_pred)), result.y_pred]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_label", "predicted_label", "confidence"])
        for image_path, true_index, predicted_index, confidence in zip(
            paths,
            result.y_true,
            result.y_pred,
            confidences,
        ):
            writer.writerow(
                [
                    image_path,
                    class_names[int(true_index)],
                    class_names[int(predicted_index)],
                    f"{float(confidence):.6f}",
                ]
            )
    return path


def save_evaluation_artifacts(
    output_dir: str | Path,
    result: EvaluationResult,
    class_names: Sequence[str],
    paths: Sequence[str],
    prefix: str = "test",
) -> dict:
    """Save metrics JSON, classification report, predictions CSV and matrix plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluation_to_dict(result, class_names)
    save_json(output_dir / f"{prefix}_metrics.json", metrics)

    report_path = output_dir / f"{prefix}_classification_report.txt"
    report_path.write_text(
        classification_report_text(result, class_names),
        encoding="utf-8",
    )

    write_predictions_csv(
        output_dir / f"{prefix}_predictions.csv",
        paths,
        result,
        class_names,
    )

    plot_confusion_matrix(
        result.confusion_matrix,
        [pretty_class_name(name) for name in class_names],
        output_dir / f"{prefix}_confusion_matrix.png",
        title=f"Confusion matrix — {prefix} set",
    )

    return metrics
