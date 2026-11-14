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

    def run_three_condition_evaluation(
        self,
        benchmarks_dir: Union[str, Path] = "data/benchmarks",
        test_dir: Union[str, Path] = "data/processed/test",
        batch_size: int = 16,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes rigorous evaluation across three fundamental forensic conditions:
          Condition 1: Clean Images (FaceForensics++, Celeb-DF, DFDC, In-distribution test)
          Condition 2: Degraded Images (JPEG recompression Q=85, 60, 35; Resizing 0.75x, 0.50x; Blur; Compound)
          Condition 3: Unseen Generator Images (Synthesized by a generator unseen during training)
        Directly measures and reports the Generalization Gap and Degradation Drop.

        Parameters:
            benchmarks_dir (Union[str, Path]): Directory containing benchmark subfolders.
            test_dir (Union[str, Path]): Directory containing test real/fake images.
            batch_size (int): Evaluation batch size.
            max_samples (Optional[int]): Maximum balanced samples per benchmark (e.g. 40 = 20 real + 20 fake).

        Returns:
            Dict[str, Any]: Comprehensive three-condition forensic evaluation metrics.
        """
        b_path = Path(benchmarks_dir)
        t_path = Path(test_dir)

        def _balance_subsample(ds: GUARDDataset, max_n: Optional[int]) -> GUARDDataset:
            if max_n is None or len(ds.samples) <= max_n:
                return ds
            half = max_n // 2
            reals = [s for s in ds.samples if s[1] == 0][:half]
            fakes = [s for s in ds.samples if s[1] == 1][:half]
            ds.samples = reals + fakes
            return ds

        report = {
            "condition_1_clean": {},
            "condition_2_degraded": {},
            "condition_3_unseen_generator": {},
            "summary_comparison": {},
        }

        # ======================================================================
        # CONDITION 1: CLEAN IMAGES (FF++, Celeb-DF, DFDC)
        # ======================================================================
        app_logger.info("=" * 70)
        app_logger.info(">>> CONDITION 1: Evaluating on CLEAN Images (Uncorrupted)")
        app_logger.info("=" * 70)

        clean_benchmarks = {
            "FaceForensics++": b_path / "FaceForensics++",
            "Celeb-DF": b_path / "Celeb-DF",
            "DFDC": b_path / "DFDC",
        }
        if t_path.exists():
            clean_benchmarks["In-Distribution-Test"] = t_path

        clean_accuracies = []
        clean_aucs = []

        for name, folder in clean_benchmarks.items():
            if folder.exists():
                ds = GUARDDataset(data_dir=folder, mode="val", target_size=(380, 380), enable_face_align=False)
                ds = _balance_subsample(ds, max_samples)
                if len(ds) > 0:
                    loader = create_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
                    metrics = self.evaluate_dataloader(loader, dataset_name=f"Clean-{name}")
                    report["condition_1_clean"][name] = metrics
                    clean_accuracies.append(metrics["accuracy"])
                    clean_aucs.append(metrics["auc_roc"])

        avg_clean_acc = float(np.mean(clean_accuracies)) if clean_accuracies else 0.0
        avg_clean_auc = float(np.mean(clean_aucs)) if clean_aucs else 0.0
        report["condition_1_clean"]["_aggregate"] = {
            "average_accuracy": avg_clean_acc,
            "average_auc_roc": avg_clean_auc,
        }

        # ======================================================================
        # CONDITION 2: DEGRADED IMAGES (JPEG sweeps, Downscaling, Blur, Compound)
        # ======================================================================
        app_logger.info("=" * 70)
        app_logger.info(">>> CONDITION 2: Evaluating on DEGRADED Images (Real-World Corruption)")
        app_logger.info("=" * 70)

        # Use in-distribution test or FF++ as source dataset for degradation testing
        sweep_source_folder = t_path if t_path.exists() else (b_path / "FaceForensics++")
        degraded_accuracies = []

        if sweep_source_folder.exists():
            base_ds = GUARDDataset(data_dir=sweep_source_folder, mode="val", target_size=(380, 380), enable_face_align=False)
            base_ds = _balance_subsample(base_ds, max_samples)

            degradation_configs = [
                ("JPEG_Q85_Mild", {"jpeg_quality": 85}),
                ("JPEG_Q60_Medium", {"jpeg_quality": 60}),
                ("JPEG_Q35_Heavy", {"jpeg_quality": 35}),
                ("Resizing_0.75x", {"downscale_scale": 0.75}),
                ("Resizing_0.50x", {"downscale_scale": 0.50}),
                ("Blur_Kernel_5", {"blur_limit": 5}),
            ]

            for deg_name, deg_params in degradation_configs:
                base_ds.augmentation.transform = get_degradation_sweep_pipeline(
                    jpeg_quality=deg_params.get("jpeg_quality"),
                    blur_limit=deg_params.get("blur_limit"),
                    downscale_scale=deg_params.get("downscale_scale"),
                    target_size=(380, 380)
                )
                loader = create_dataloader(base_ds, batch_size=batch_size, shuffle=False, num_workers=0)
                metrics = self.evaluate_dataloader(loader, dataset_name=f"Degraded-{deg_name}")
                report["condition_2_degraded"][deg_name] = metrics
                degraded_accuracies.append(metrics["accuracy"])

        # Check for pre-rendered degraded test folder
        degraded_folder = Path("data/processed/test_degraded")
        if degraded_folder.exists():
            ds_deg = GUARDDataset(data_dir=degraded_folder, mode="val", target_size=(380, 380), enable_face_align=False)
            ds_deg = _balance_subsample(ds_deg, max_samples)
            if len(ds_deg) > 0:
                loader = create_dataloader(ds_deg, batch_size=batch_size, shuffle=False, num_workers=0)
                metrics = self.evaluate_dataloader(loader, dataset_name="Compound_JPEG50_Resize0.7x")
                report["condition_2_degraded"]["Compound_JPEG50_Resize0.7x"] = metrics
                degraded_accuracies.append(metrics["accuracy"])

        avg_degraded_acc = float(np.mean(degraded_accuracies)) if degraded_accuracies else 0.0
        report["condition_2_degraded"]["_aggregate"] = {
            "average_accuracy": avg_degraded_acc,
            "degradation_drop": avg_clean_acc - avg_degraded_acc,
        }

        # ======================================================================
        # CONDITION 3: UNSEEN GENERATOR IMAGES (Generalization Gap)
        # ======================================================================
        app_logger.info("=" * 70)
        app_logger.info(">>> CONDITION 3: Evaluating on UNSEEN GENERATOR Images")
        app_logger.info("=" * 70)

        unseen_folder = b_path / "Unseen_Generator"
        unseen_acc = 0.0
        unseen_auc = 0.0

        if unseen_folder.exists():
            ds_unseen = GUARDDataset(data_dir=unseen_folder, mode="val", target_size=(380, 380), enable_face_align=False)
            ds_unseen = _balance_subsample(ds_unseen, max_samples)
            if len(ds_unseen) > 0:
                loader = create_dataloader(ds_unseen, batch_size=batch_size, shuffle=False, num_workers=0)
                metrics = self.evaluate_dataloader(loader, dataset_name="Unseen-Generator-Benchmark")
                report["condition_3_unseen_generator"]["Unseen_Generator"] = metrics
                unseen_acc = metrics["accuracy"]
                unseen_auc = metrics["auc_roc"]

        generalization_gap = avg_clean_acc - unseen_acc

        report["summary_comparison"] = {
            "condition_1_clean_accuracy": avg_clean_acc,
            "condition_1_clean_auc": avg_clean_auc,
            "condition_2_degraded_accuracy": avg_degraded_acc,
            "condition_2_degradation_drop": avg_clean_acc - avg_degraded_acc,
            "condition_3_unseen_generator_accuracy": unseen_acc,
            "condition_3_unseen_generator_auc": unseen_auc,
            "generalization_gap": generalization_gap,
        }

        app_logger.info("=" * 70)
        app_logger.info(">>> THREE-CONDITION EVALUATION COMPLETE")
        app_logger.info(f"    - Clean Images Accuracy:            {avg_clean_acc * 100:.2f}% (AUC: {avg_clean_auc:.4f})")
        app_logger.info(f"    - Degraded Images Accuracy:         {avg_degraded_acc * 100:.2f}% (Drop: {(avg_clean_acc - avg_degraded_acc) * 100:.2f}%)")
        app_logger.info(f"    - Unseen Generator Accuracy:        {unseen_acc * 100:.2f}% (AUC: {unseen_auc:.4f})")
        app_logger.info(f"    - Visible Generalization Gap:       {generalization_gap * 100:.2f}%")
        app_logger.info("=" * 70)

        return report

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
