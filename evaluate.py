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
        default="three_conditions",
        choices=["three_conditions", "in_distribution", "cross_dataset", "degradation_sweep"],
        help="Evaluation protocol mode to execute."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/guard_model.pth",
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum balanced samples per benchmark evaluated (e.g. 40 = 20 real + 20 fake)."
    )

    return parser.parse_args()


def print_formatted_results(report: dict) -> None:
    """
    Renders an ASCII summary table for the three forensic evaluation conditions.
    """
    print("\n" + "=" * 82)
    print(" GUARD FORENSIC EVALUATION BENCHMARK REPORT (MEASURED RESULTS)")
    print("=" * 82)

    # Condition 1
    c1 = report.get("condition_1_clean", {})
    print("\n [CONDITION 1: CLEAN IMAGES (UNCORRUPTED)]")
    print(f" {'Benchmark / Dataset':<28} | {'Accuracy':<10} | {'AUC-ROC':<10} | {'EER':<10} | {'F1-Score':<10}")
    print(" " + "-" * 80)
    for name, metrics in c1.items():
        if name.startswith("_"):
            continue
        print(f" {name:<28} | {metrics['accuracy']*100:6.2f}%    | {metrics['auc_roc']:8.4f}   | {metrics['eer']*100:6.2f}%    | {metrics['f1_score']:8.4f}")
    c1_agg = c1.get("_aggregate", {})
    if c1_agg:
        print(" " + "-" * 80)
        print(f" {'Average Clean Aggregate':<28} | {c1_agg['average_accuracy']*100:6.2f}%    | {c1_agg['average_auc_roc']:8.4f}   | {'--':<10} | {'--':<10}")

    # Condition 2
    c2 = report.get("condition_2_degraded", {})
    print("\n [CONDITION 2: DEGRADED IMAGES (REAL-WORLD CORRUPTIONS)]")
    print(f" {'Degradation Perturbation':<28} | {'Accuracy':<10} | {'AUC-ROC':<10} | {'EER':<10} | {'F1-Score':<10}")
    print(" " + "-" * 80)
    for name, metrics in c2.items():
        if name.startswith("_"):
            continue
        print(f" {name:<28} | {metrics['accuracy']*100:6.2f}%    | {metrics['auc_roc']:8.4f}   | {metrics['eer']*100:6.2f}%    | {metrics['f1_score']:8.4f}")
    c2_agg = c2.get("_aggregate", {})
    if c2_agg:
        print(" " + "-" * 80)
        print(f" {'Average Degraded Accuracy':<28} | {c2_agg['average_accuracy']*100:6.2f}%    | {'Drop:':<8} {c2_agg['degradation_drop']*100:5.2f}%")

    # Condition 3
    c3 = report.get("condition_3_unseen_generator", {})
    summary = report.get("summary_comparison", {})
    print("\n [CONDITION 3: UNSEEN GENERATOR & GENERALIZATION GAP]")
    print(f" {'Generator Architecture':<28} | {'Accuracy':<10} | {'AUC-ROC':<10} | {'EER':<10} | {'F1-Score':<10}")
    print(" " + "-" * 80)
    for name, metrics in c3.items():
        if name.startswith("_"):
            continue
        print(f" {name:<28} | {metrics['accuracy']*100:6.2f}%    | {metrics['auc_roc']:8.4f}   | {metrics['eer']*100:6.2f}%    | {metrics['f1_score']:8.4f}")
    print(" " + "-" * 80)
    print(f" Generalization Gap (Clean In-Dist vs Unseen Gen): {summary.get('generalization_gap', 0.0)*100:+.2f}%")
    print("=" * 82 + "\n")


def main() -> None:
    """
    Main evaluation routine.
    """
    args = parse_args()
    print_device_summary()
    device = get_device()

    # Checkpoint fallback handling
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        fallback_best = Path("checkpoints/guard_best.pth")
        if fallback_best.exists():
            ckpt_path = fallback_best
        else:
            app_logger.warning(f"No checkpoint found at {ckpt_path}. Model will evaluate with initialized weights.")

    # 1. Initialize master GUARD architecture
    model = GUARDModel()
    evaluator = GUARDEvaluator(
        model=model,
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        device=device
    )

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Execute selected evaluation protocol
    if args.mode == "three_conditions":
        results = evaluator.run_three_condition_evaluation(
            benchmarks_dir=args.benchmarks_dir,
            test_dir=args.data_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        evaluator.save_report(results, out_path)
        print_formatted_results(results)

    elif args.mode == "in_distribution":
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

        loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        results = evaluator.evaluate_dataloader(loader, dataset_name="In-Distribution-Test")
        evaluator.save_report(results, out_path)

    elif args.mode == "cross_dataset":
        bench_path = Path(args.benchmarks_dir)
        benchmarks = {
            "FaceForensics++": bench_path / "FaceForensics++",
            "Celeb-DF": bench_path / "Celeb-DF",
            "DFDC": bench_path / "DFDC",
            "Unseen_Generator": bench_path / "Unseen_Generator",
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
