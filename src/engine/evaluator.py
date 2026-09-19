# ==============================================================================
# File: src/engine/evaluator.py
# Description: Rigorous Evaluation Protocol Engine for GUARD.
#
# Protocols Implemented:
#   1. In-Distribution Evaluation (FaceForensics++ test partition).
#   2. Cross-Dataset Generalization Testing (Celeb-DF-v2, DFDC, DiffusionForensics).
#   3. Robustness Degradation Sweeps (Quantization JPEG 90-30, Optical Blurs, Downsampling).
# ==============================================================================

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import GuardConfig, default_config
from src.models.guard_model import GUARDModel
from src.data.dataset import GUARDDataset, create_dataloader
from src.data.augmentations import get_degradation_sweep_pipeline
from src.utils.device import get_device
from src.utils.logger import app_logger
from src.utils.metrics import evaluate_predictions


class GUARDEvaluator:
    """
    Standardized benchmark evaluation suite for deepfake and generative image detection.
    Executes in-distribution validation, cross-dataset generalization testing,
    and single-factor degradation robustness sweeps.
    """

    def __init__(
        self,
        model: GUARDModel,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[GuardConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters:
            model (GUARDModel): Model instance to evaluate.
            checkpoint_path (Optional[Union[str, Path]]): Optional weights file to load.
            config (Optional[GuardConfig]): Master configuration.
            device (Optional[torch.device]): Evaluation device (CUDA/CPU).
        """
        self.config = config or default_config
        self.device = device or get_device()
        self.model = model.to(self.device)

        # If a pre-trained checkpoint path is provided, load weights
        if checkpoint_path is not None:
            self.load_checkpoint(Path(checkpoint_path))

        self.model.eval()

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Loads saved model weights from disk.

        Parameters:
            checkpoint_path (Path): Path to saved .pth file.
        """
        if not checkpoint_path.exists():
            app_logger.warning(f"Checkpoint not found at: {checkpoint_path}. Using current model weights.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        app_logger.info(f"Loaded checkpoint weights from: {checkpoint_path}")

    @torch.no_grad()
    def evaluate_dataloader(
        self,
        data_loader: DataLoader,
        dataset_name: str = "Test-Set"
    ) -> Dict[str, Any]:
        """
        Runs evaluation over a PyTorch DataLoader and computes complete metrics.

        Parameters:
            data_loader (DataLoader): DataLoader instance.
            dataset_name (str): Human-readable name for the benchmark dataset.

        Returns:
            Dict[str, Any]: Comprehensive metrics dictionary.
        """
        self.model.eval()
        all_labels = []
        all_probs = []
        all_paths = []

        app_logger.info(f"Evaluating benchmark: {dataset_name} ({len(data_loader.dataset)} samples)...")

        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            outputs = self.model(images)
            probs = outputs["probabilities"]

            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_paths.extend(batch["path"])

        y_true = np.vstack(all_labels).flatten()
        y_scores = np.vstack(all_probs).flatten()

        metrics = evaluate_predictions(y_true, y_scores)
        metrics["dataset_name"] = dataset_name
        metrics["total_samples"] = len(y_true)

        app_logger.info(
            f"[{dataset_name}] Accuracy: {metrics['accuracy'] * 100:.2f}% | "
            f"AUC-ROC: {metrics['auc_roc']:.4f} | "
            f"EER: {metrics['eer'] * 100:.2f}% | "
            f"F1: {metrics['f1_score']:.4f}"
        )

        return metrics

    def run_cross_dataset_evaluation(
        self,
        dataset_paths: Dict[str, Union[str, Path]],
        batch_size: int = 16
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes cross-dataset evaluation across multiple independent benchmark datasets
        (e.g., FaceForensics++, Celeb-DF-v2, DFDC, DiffusionForensics) to measure
        the cross-domain generalization gap.

        Parameters:
            dataset_paths (Dict[str, Path]): Dict mapping benchmark names to their folder paths.
            batch_size (int): Evaluation batch size.

        Returns:
            Dict[str, Dict[str, Any]]: Evaluation results keyed by dataset name.
        """
        results = {}

        for ds_name, ds_path in dataset_paths.items():
            path_obj = Path(ds_path)
            if not path_obj.exists():
                app_logger.warning(f"Benchmark path does not exist, skipping: {path_obj}")
                continue

            # Instantiate dataset in validation mode (deterministic resizing, no random augmentations)
            ds = GUARDDataset(data_dir=path_obj, mode="val", target_size=(380, 380))
            if len(ds) == 0:
                app_logger.warning(f"Benchmark directory is empty: {path_obj}")
                continue

            loader = create_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
            metrics = self.evaluate_dataloader(loader, dataset_name=ds_name)
            results[ds_name] = metrics

        return results

    def run_degradation_sweep(
        self,
        dataset: GUARDDataset,
        jpeg_qualities: Optional[List[int]] = None,
        blur_limits: Optional[List[int]] = None,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Sweeps through controlled single-factor degradations (JPEG compression levels, blur kernels)
        to evaluate model resilience against social media degradation.

        Parameters:
            dataset (GUARDDataset): Source evaluation dataset.
            jpeg_qualities (Optional[List[int]]): List of JPEG quality factors (e.g. [90, 70, 50, 30]).
            blur_limits (Optional[List[int]]): List of blur kernel limits (e.g. [3, 5, 7]).
            batch_size (int): Batch size.

        Returns:
            Dict[str, Any]: Mapping of degradation factors to resulting performance metrics.
        """
        sweep_results = {
            "jpeg_sweep": {},
            "blur_sweep": {},
        }

        # 1. JPEG Compression Sweep
        qualities = jpeg_qualities or self.config.evaluation.jpeg_sweep_qualities
        app_logger.info(f"Starting JPEG Degradation Sweep across qualities: {qualities}")

        for q in qualities:
            # Overwrite active augmentation transform with fixed JPEG degradation
            dataset.augmentation.transform = get_degradation_sweep_pipeline(
                jpeg_quality=q,
                target_size=dataset.target_size
            )
            loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            metrics = self.evaluate_dataloader(loader, dataset_name=f"JPEG-Q{q}")
            sweep_results["jpeg_sweep"][f"Q_{q}"] = metrics

        # 2. Blur Sweep
        blurs = blur_limits or [3, 5, 7]
        app_logger.info(f"Starting Blur Degradation Sweep across kernel limits: {blurs}")

        for b in blurs:
            dataset.augmentation.transform = get_degradation_sweep_pipeline(
                blur_limit=b,
                target_size=dataset.target_size
            )
            loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            metrics = self.evaluate_dataloader(loader, dataset_name=f"Blur-K{b}")
            sweep_results["blur_sweep"][f"Kernel_{b}"] = metrics

        return sweep_results

    def save_report(
        self,
        report_data: Dict[str, Any],
        output_filepath: Union[str, Path]
    ) -> None:
        """
        Exports evaluation metrics to a structured JSON file on disk.

        Parameters:
            report_data (Dict[str, Any]): Metrics report dictionary.
            output_filepath (Union[str, Path]): Target output file path.
        """
        out_path = Path(output_filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)

        app_logger.info(f"Saved evaluation report to: {out_path}")
