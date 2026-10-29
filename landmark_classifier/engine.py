"""Training loop, validation and evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .model import set_frozen_batchnorm_eval


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    confusion_matrix: np.ndarray
    report: dict


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    if training:
        set_frozen_batchnorm_eval(model)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size

    if total == 0:
        raise RuntimeError("The data loader is empty.")

    return EpochMetrics(loss=total_loss / total, accuracy=correct / total)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    phase: str,
    patience: int = 0,
    logger: Callable[[str], None] = print,
) -> tuple[list[dict], float, dict[str, torch.Tensor], int]:
    """Train for up to ``epochs`` epochs and return the best validation state.

    The returned state dict is stored on CPU so it does not occupy GPU memory.
    ``best_val_loss`` is the validation loss of the restored checkpoint.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if patience < 0:
        raise ValueError("patience must be >= 0.")

    history: list[dict] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)

        if not math.isfinite(val_metrics.loss):
            raise RuntimeError(
                f"Validation loss became non-finite in phase '{phase}', epoch {epoch}."
            )

        learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "phase": phase,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
        }
        history.append(record)

        logger(
            f"[{phase}] epoch {epoch:02d}/{epochs:02d} | "
            f"train loss {train_metrics.loss:.4f} | "
            f"train acc {train_metrics.accuracy:.4f} | "
            f"val loss {val_metrics.loss:.4f} | "
            f"val acc {val_metrics.accuracy:.4f} | "
            f"lr {learning_rate:.1e}"
        )

        if val_metrics.loss < best_val_loss - 1e-6:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            logger(
                f"[{phase}] early stopping after {epoch} epochs "
                f"(no validation improvement for {patience} epochs)."
            )
            break

    if best_state is None:
        # Defensive fallback: should only happen if the first validation loss
        # was NaN, which is handled above.
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_val_loss = history[-1]["val_loss"]
        best_epoch = history[-1]["epoch"]

    model.load_state_dict(best_state)
    return history, best_val_loss, best_state, best_epoch


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> EvaluationResult:
    """Run the model on a loader and compute loss, accuracy, F1 and confusion matrix."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((predictions == targets).sum().item())
        total += batch_size

        all_probabilities.append(probabilities.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(targets.cpu().numpy())

    if total == 0:
        raise RuntimeError("The evaluation loader is empty.")

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    probabilities_array = np.concatenate(all_probabilities)

    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return EvaluationResult(
        loss=total_loss / total,
        accuracy=correct / total,
        macro_f1=float(report["macro avg"]["f1-score"]),
        weighted_f1=float(report["weighted avg"]["f1-score"]),
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities_array,
        confusion_matrix=matrix,
        report=report,
    )
