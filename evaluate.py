# ==============================================================================
# File: evaluate.py
# Description: Command-Line Evaluation Protocol Entry Point for GUARD.
#
# Protocols:
#   1. In-Distribution Evaluation (FF++ test set).
#   2. Cross-Dataset Generalization (Celeb-DF-v2, DFDC, DiffusionForensics).
#   3. Degradation Sweeps (JPEG compression and blur robustness curves).
#
# Usage Example:
#   python evaluate.py --mode in_distribution --data-dir data/processed/test
#   python evaluate.py --mode degradation_sweep --data-dir data/processed/test
# ==============================================================================

import argparse
import sys
from pathlib import Path
import torch

from configs.config import default_config
from src.models.guard_model import GUARDModel
from src.data.dataset import GUARDDataset, create_dataloader
from src.engine.evaluator import GUARDEvaluator
from src.utils.device import get_device, print_device_summary
from src.utils.logger import app_logger


def parse_args() -> argparse.Namespace:
    """
    Parses command-line execution flags for evaluation protocols.
    """
    parser = argparse.ArgumentParser(
        description="GUARD: Standardized Evaluation and Robustness Benchmark Protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="in_distribution",
        choices=["in_distribution", "cross_dataset", "degradation_sweep"],
        help="Evaluation protocol mode to execute."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/guard_best.pth",
        help="Path to saved model checkpoint weights (.pth)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/test",
        help="Path to evaluation dataset directory."
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=str,
        default="data/benchmarks",
        help="Directory containing benchmark subfolders for cross-dataset evaluation."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="outputs/evaluation_report.json",
        help="Destination path for JSON metrics report."
    )

    return parser.parse_args()


def main() -> None:
    """
    Main evaluation routine.
    """
    args = parse_args()
    print_device_summary()
    device = get_device()

    # 1. Initialize master GUARD architecture
    model = GUARDModel()
    evaluator = GUARDEvaluator(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device
    )

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Execute selected evaluation protocol
    if args.mode == "in_distribution":
        data_path = Path(args.data_dir)
        if not data_path.exists():
            app_logger.warning(
                f"Evaluation dataset directory not found at: {data_path}. "
                f"Please populate images in 'data/processed/test/real' and 'data/processed/test/fake'."
            )
            sys.exit(0)

        dataset = GUARDDataset(data_dir=data_path, mode="test", target_size=(380, 380))
        if len(dataset) == 0:
            app_logger.warning(f"No valid images discovered in: {data_path}")
            sys.exit(0)

        loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        results = evaluator.evaluate_dataloader(loader, dataset_name="In-Distribution-Test")
        evaluator.save_report(results, out_path)

    elif args.mode == "cross_dataset":
        bench_path = Path(args.benchmarks_dir)
        benchmarks = {
            "FaceForensics++": bench_path / "FaceForensics++",
            "Celeb-DF-v2": bench_path / "Celeb-DF-v2",
            "DFDC": bench_path / "DFDC",
            "DiffusionForensics": bench_path / "DiffusionForensics",
        }
        results = evaluator.run_cross_dataset_evaluation(benchmarks, batch_size=args.batch_size)
        evaluator.save_report(results, out_path)

    elif args.mode == "degradation_sweep":
        data_path = Path(args.data_dir)
        if not data_path.exists():
            app_logger.warning(f"Dataset path for degradation sweep not found: {data_path}")
            sys.exit(0)

        dataset = GUARDDataset(data_dir=data_path, mode="test", target_size=(380, 380))
        if len(dataset) == 0:
            app_logger.warning(f"No images found for sweep in: {data_path}")
            sys.exit(0)

        results = evaluator.run_degradation_sweep(dataset, batch_size=args.batch_size)
        evaluator.save_report(results, out_path)

    app_logger.info(f"Evaluation complete. Full report written to: {out_path}")


if __name__ == "__main__":
    main()
