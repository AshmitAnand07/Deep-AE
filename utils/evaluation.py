"""
evaluation.py
=============
Deep-AE Phase 1 — Evaluation Utilities

This module provides functions for computing classification metrics
and generating a confusion matrix visualization after training.

Metrics computed:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix (saved as PNG)

Author: Deep-AE Team
"""

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_loader import INDEX_TO_CLASS

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, object]:
    """
    Runs inference on a DataLoader and computes classification metrics.

    Args:
        model (nn.Module): Trained model in eval mode.
        data_loader (DataLoader): Test or validation DataLoader.
        device (torch.device): Device to run inference on.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, and
              the raw lists of all_labels and all_preds for confusion matrix.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Convert to numpy arrays for calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Compute Metrics ---
    num_classes = len(INDEX_TO_CLASS)
    accuracy = np.mean(all_preds == all_labels)

    # Per-class precision, recall, F1
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for cls_idx in range(num_classes):
        tp = np.sum((all_preds == cls_idx) & (all_labels == cls_idx))
        fp = np.sum((all_preds == cls_idx) & (all_labels != cls_idx))
        fn = np.sum((all_preds != cls_idx) & (all_labels == cls_idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    # Macro averages
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    results = {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "all_labels": all_labels,
        "all_preds": all_preds,
    }

    return results


def print_metrics(results: Dict[str, object]) -> None:
    """
    Prints a formatted summary of evaluation metrics.

    Args:
        results (dict): Output from evaluate_model().
    """
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Accuracy  : {results['accuracy']:.4f}")
    logger.info(f"  Precision : {results['precision']:.4f} (macro)")
    logger.info(f"  Recall    : {results['recall']:.4f} (macro)")
    logger.info(f"  F1 Score  : {results['f1']:.4f} (macro)")
    logger.info("-" * 50)

    for cls_idx, cls_name in INDEX_TO_CLASS.items():
        logger.info(
            f"  [{cls_name}]  "
            f"Precision={results['precision_per_class'][cls_idx]:.4f}  "
            f"Recall={results['recall_per_class'][cls_idx]:.4f}  "
            f"F1={results['f1_per_class'][cls_idx]:.4f}"
        )
    logger.info("=" * 50)


def plot_confusion_matrix(
    results: Dict[str, object],
    save_path: str = "models/confusion_matrix.png"
) -> None:
    """
    Generates and saves a confusion matrix visualization.

    Args:
        results (dict): Output from evaluate_model().
        save_path (str): Path to save the confusion matrix image.
    """
    all_labels = results["all_labels"]
    all_preds = results["all_preds"]
    num_classes = len(INDEX_TO_CLASS)
    class_names = [INDEX_TO_CLASS[i] for i in range(num_classes)]

    # Build the confusion matrix manually (no sklearn dependency needed)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        matrix[true][pred] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix — Deep-AE CNN", fontsize=14, pad=15)
    fig.colorbar(im, ax=ax)

    # Labels
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # Annotate each cell with the count
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center", fontsize=16, color=color, fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to: {save_path}")


def plot_training_history(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str = "models/training_curves.png"
) -> None:
    """
    Plots and saves training/validation loss and accuracy curves.

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        train_accs (list): Training accuracy per epoch.
        val_accs (list): Validation accuracy per epoch.
        save_path (str): Path to save the plot.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Deep-AE CNN Training History", fontsize=15, fontweight="bold")

    # Loss curve
    ax1.plot(epochs, train_losses, "o-", label="Train Loss", color="#e74c3c", markersize=4)
    ax1.plot(epochs, val_losses, "o-", label="Val Loss", color="#3498db", markersize=4)
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, train_accs, "o-", label="Train Accuracy", color="#e74c3c", markersize=4)
    ax2.plot(epochs, val_accs, "o-", label="Val Accuracy", color="#3498db", markersize=4)
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved to: {save_path}")
