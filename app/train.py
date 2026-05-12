"""
train.py
========
Deep-AE Phase 1 — Complete CNN Training Pipeline

This is the main entry point for training the DeepAE_CNN model.

Pipeline:
  1. Set random seeds for reproducibility.
  2. Detect and select hardware device (GPU or CPU).
  3. Load the spectrogram dataset and split into train/val/test.
  4. Initialize the CNN model, optimizer, and loss function.
  5. Run the training loop with validation at each epoch.
  6. Save the best model checkpoint based on validation accuracy.
  7. Evaluate the best model on the test set.
  8. Generate confusion matrix and training curve visualizations.

Usage:
  python train.py

Author: Deep-AE Team
"""

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add root directory to python path for package imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.dataset_loader import get_data_loaders, INDEX_TO_CLASS
from models.model import DeepAE_CNN, get_model_summary
from utils.evaluation import (
    evaluate_model,
    print_metrics,
    plot_confusion_matrix,
    plot_training_history,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Training hyperparameters
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_CLASSES = 2
SEED = 42

# Paths
DATA_DIR = "dataset/spectrograms"
MODEL_SAVE_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "deepae_cnn.pth")

# ============================================================
# 2. LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:
    """
    Sets all random seeds for full reproducibility across runs.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


# ============================================================
# 4. TRAINING LOOP (one epoch)
# ============================================================

def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """
    Runs one full training epoch.

    Args:
        model: The CNN model.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU or GPU.

    Returns:
        tuple: (average_loss, accuracy) for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============================================================
# 5. VALIDATION LOOP (one epoch)
# ============================================================

def validate_one_epoch(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Runs one full validation epoch (no gradient computation).

    Args:
        model: The CNN model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: CPU or GPU.

    Returns:
        tuple: (average_loss, accuracy) for this epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


# ============================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("  Deep-AE CNN Training Pipeline")
    logger.info("=" * 60)

    # --- Step 1: Reproducibility ---
    set_seed(SEED)

    # --- Step 2: Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Step 3: Load Data ---
    logger.info(f"Loading dataset from: {DATA_DIR}")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # --- Step 4: Initialize Model ---
    model = DeepAE_CNN(num_classes=NUM_CLASSES).to(device)
    logger.info(f"Model Architecture:\n{model}")
    logger.info(f"\n{get_model_summary(model)}")

    # --- Step 5: Loss Function & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info(f"Optimizer  : Adam (lr={LEARNING_RATE})")
    logger.info(f"Loss       : CrossEntropyLoss")
    logger.info(f"Epochs     : {EPOCHS}")
    logger.info(f"Batch Size : {BATCH_SIZE}")

    # --- Step 6: Training Loop ---
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    logger.info("-" * 60)
    logger.info("Starting Training...")
    logger.info("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # Validate
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Log epoch results
        logger.info(
            f"Epoch [{epoch:02d}/{EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "hyperparameters": {
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                },
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            logger.info(f"  [BEST] New best model saved! (Val Acc: {val_acc:.4f})")

    logger.info("-" * 60)
    logger.info(f"Training Complete. Best Val Accuracy: {best_val_acc:.4f}")
    logger.info("-" * 60)

    # --- Step 7: Generate Training Curves ---
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(MODEL_SAVE_DIR, "training_curves.png"),
    )

    # --- Step 8: Load Best Model & Evaluate on Test Set ---
    logger.info("Loading best model for final evaluation on test set...")
    best_checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_results = evaluate_model(model, test_loader, device)
    print_metrics(test_results)

    # --- Step 9: Confusion Matrix ---
    plot_confusion_matrix(
        test_results,
        save_path=os.path.join(MODEL_SAVE_DIR, "confusion_matrix.png"),
    )

    logger.info("=" * 60)
    logger.info("  Pipeline finished successfully.")
    logger.info(f"  Model saved at     : {MODEL_SAVE_PATH}")
    logger.info(f"  Training curves at : {MODEL_SAVE_DIR}/training_curves.png")
    logger.info(f"  Confusion matrix at: {MODEL_SAVE_DIR}/confusion_matrix.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
