"""Regenerate the result figures from the recorded confusion matrices.

The original training run produced the raw figures in the university report.
This script keeps only the matrix-backed numbers and renders clean, English,
portfolio-friendly figures. It has no PyTorch dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from landmark_classifier.plots import plot_confusion_matrix, pretty_class_name

MODELS = [
    ("mobilenetv3", "MobileNetV3-Large 100"),
    ("efficientnet_lite0", "EfficientNet-Lite0"),
]


def compute_metrics(confusion: np.ndarray) -> dict:
    confusion = np.asarray(confusion, dtype=np.int64)
    row_totals = confusion.sum(axis=1)
    column_totals = confusion.sum(axis=0)

    recall = np.divide(
        np.diag(confusion),
        row_totals,
        out=np.zeros(len(confusion), dtype=np.float64),
        where=row_totals > 0,
    )
    precision = np.divide(
        np.diag(confusion),
        column_totals,
        out=np.zeros(len(confusion), dtype=np.float64),
        where=column_totals > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(confusion), dtype=np.float64),
        where=(precision + recall) > 0,
    )

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "images")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for key, display_name in MODELS:
        path = args.results_dir / f"{key}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        data = load_result(path)
        matrix = np.asarray(data["confusion_matrix"], dtype=np.int64)
        metrics = compute_metrics(matrix)
        loaded.append((key, display_name, data, matrix, metrics))

        plot_confusion_matrix(
            matrix,
            [pretty_class_name(name) for name in data["class_names"]],
            args.output_dir / f"confusion_matrix_{key}.png",
            title=f"{display_name} — {metrics['total']}-image test set",
        )

    print("| Model | Correct | Accuracy | Macro recall | Macro F1 |")
    print("|---|---:|---:|---:|---:|")
    for _, display_name, _, _, metrics in loaded:
        print(
            f"| {display_name} | {metrics['correct']}/{metrics['total']} | "
            f"{metrics['accuracy']:.1%} | {metrics['macro_recall']:.1%} | "
            f"{metrics['macro_f1']:.1%} |"
        )

    class_names = [pretty_class_name(name) for name in loaded[0][2]["class_names"]]
    print("\nPer-class recall:")
    print("| Class | MobileNetV3-Large | EfficientNet-Lite0 |")
    print("|---|---:|---:|")
    for index, name in enumerate(class_names):
        print(
            f"| {name} | {loaded[0][4]['recall'][index]:.1%} | "
            f"{loaded[1][4]['recall'][index]:.1%} |"
        )

    x = np.arange(len(class_names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        x - width / 2,
        loaded[0][4]["recall"],
        width,
        label="MobileNetV3-Large 100",
        color="#1f77b4",
    )
    axis.bar(
        x + width / 2,
        loaded[1][4]["recall"],
        width,
        label="EfficientNet-Lite0",
        color="#ff7f0e",
    )
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Recall")
    axis.set_title("Per-class recall on the 95-image test set")
    axis.set_xticks(x)
    axis.set_xticklabels(class_names, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "per_class_recall.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"\nFigures written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
