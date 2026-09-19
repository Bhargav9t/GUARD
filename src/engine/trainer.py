# ==============================================================================
# File: src/engine/trainer.py
# Description: Training Engine for the GUARD Architecture.
# Manages optimization loops, mixed precision (AMP), gradient clipping,
# cosine learning rate scheduling, validation checkpoints, and metric logging.
# ==============================================================================

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import GuardConfig, default_config
from src.models.guard_model import GUARDModel
from src.utils.device import get_device
from src.utils.logger import app_logger
from src.utils.metrics import evaluate_predictions


class GUARDTrainer:
    """
    Dedicated Training Orchestrator for GUARD.
    Supports end-to-end multi-modal optimization, automated mixed precision (AMP),
    gradient clipping, and best-model checkpointing based on AUC-ROC and EER.
    """

    def __init__(
        self,
        model: GUARDModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[GuardConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters:
            model (GUARDModel): Master GUARD neural network.
            train_loader (DataLoader): DataLoader for training split.
            val_loader (Optional[DataLoader]): DataLoader for validation split.
            config (Optional[GuardConfig]): Master configuration.
            device (Optional[torch.device]): Target hardware device (CUDA/CPU).
        """
        self.config = config or default_config
        self.device = device or get_device()
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # 1. Binary Cross Entropy with Logits Loss (numerically stable for raw model logits)
        self.criterion = nn.BCEWithLogitsLoss()

        # 2. AdamW Optimizer with decoupled weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # 3. Cosine Annealing Learning Rate Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.epochs,
            eta_min=self.config.training.min_lr,
        )

        # 4. Automated Mixed Precision (AMP) Scaler for accelerated CUDA training
        self.use_amp = self.config.training.mixed_precision and torch.cuda.is_available()
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 5. Checkpoint destination directory
        self.checkpoints_dir = Path(self.config.paths.checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Tracking state
        self.best_val_auc = 0.0
        self.best_val_eer = 1.0

        app_logger.info(
            f"GUARDTrainer initialized on {self.device}: "
            f"Epochs={self.config.training.epochs}, LR={self.config.training.learning_rate}, "
            f"AMP={self.use_amp}"
        )

    def train_epoch(self, epoch: int) -> float:
        """
        Executes one full training epoch across the training dataset.

        Parameters:
            epoch (int): Current epoch index (1-indexed).

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        batch_count = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.training.epochs} [Train]")

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Execute forward pass with Automated Mixed Precision
            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=self.use_amp)
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )
            with autocast_ctx:
                outputs = self.model(images)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Unscale gradients prior to gradient norm clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.max_grad_norm
            )

            # Optimizer step and scaler update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update progress bar display
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "LR": f"{current_lr:.2e}"})

        return total_loss / max(1, batch_count)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Evaluates the model on the validation split and computes quantitative metrics.

        Returns:
            Dict[str, float]: Dictionary of validation metrics (loss, accuracy, auc_roc, eer).
        """
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}

        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=self.use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else torch.cuda.amp.autocast(enabled=self.use_amp)
        )

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast_ctx:
                outputs = self.model(images)
                logits = outputs["logits"]
                probs = outputs["probabilities"]
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

        avg_loss = total_loss / max(1, len(self.val_loader))
        y_true = np.vstack(all_labels).flatten()
        y_scores = np.vstack(all_probs).flatten()

        # Compute forensic metrics (AUC-ROC, EER, F1, Accuracy)
        metrics = evaluate_predictions(y_true, y_scores)
        metrics["val_loss"] = avg_loss

        return metrics

    def save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Serializes model parameters, optimizer state, scheduler, and metrics.

        Parameters:
            filepath (Path): Destination .pth file path.
            epoch (int): Current epoch number.
            metrics (Dict[str, float]): Validation metrics dictionary.
            is_best (bool): Whether this checkpoint achieved top validation score.
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint_data, filepath)
        app_logger.info(f"Saved checkpoint to: {filepath}")

        # If best model, also save to dedicated 'guard_best.pth'
        if is_best:
            best_path = self.checkpoints_dir / "guard_best.pth"
            torch.save(checkpoint_data, best_path)
            app_logger.info(f"*** New best checkpoint recorded at: {best_path} ***")

    def fit(self) -> Dict[str, Any]:
        """
        Executes the full training and validation lifecycle over the configured epochs.

        Returns:
            Dict[str, Any]: Training history containing per-epoch loss and metrics.
        """
        epochs = self.config.training.epochs
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_eer": [],
        }

        app_logger.info(f"Starting GUARD training run for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # 1. Execute training epoch
            train_loss = self.train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # 2. Step learning rate scheduler
            self.scheduler.step()

            # 3. Validate performance on validation split
            if self.val_loader is not None:
                val_metrics = self.validate()
                val_loss = val_metrics.get("val_loss", 0.0)
                val_auc = val_metrics.get("auc_roc", 0.0)
                val_eer = val_metrics.get("eer", 1.0)

                history["val_loss"].append(val_loss)
                history["val_auc"].append(val_auc)
                history["val_eer"].append(val_eer)

                app_logger.info(
                    f"Epoch {epoch:02d}/{epochs:02d} - "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | Val EER: {val_eer:.4f}"
                )

                # Check if current epoch is the best performing model
                is_best = val_auc > self.best_val_auc
                if is_best:
                    self.best_val_auc = val_auc
                    self.best_val_eer = val_eer

                # Save periodic or best checkpoint
                latest_path = self.checkpoints_dir / "guard_latest.pth"
                self.save_checkpoint(latest_path, epoch, val_metrics, is_best=is_best)

                if epoch % self.config.training.save_frequency == 0:
                    epoch_path = self.checkpoints_dir / f"guard_epoch_{epoch:03d}.pth"
                    self.save_checkpoint(epoch_path, epoch, val_metrics, is_best=False)

        elapsed = time.time() - start_time
        app_logger.info(f"Training completed in {elapsed / 60:.2f} minutes. Best Val AUC: {self.best_val_auc:.4f}")

        return history
