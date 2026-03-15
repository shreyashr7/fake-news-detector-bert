from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import Day5Config
from src.dataset import FakeNewsDataset
from src.device import get_device_info, get_best_device
from src.evaluator import (
    build_confusion_outputs,
    collect_predictions,
    compute_metrics,
    save_classification_report,
    save_confusion_matrix_plot,
    save_summary_json,
    save_threshold_analysis,
)


def parse_args() -> Day5Config:
    parser = argparse.ArgumentParser(description="Day 5: Formal model evaluation and baseline reporting")
    parser.add_argument("--eval-csv", type=Path, default=Path("artifacts/data_splits/val.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model_improved/best_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation_optimized"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    return Day5Config(
        eval_csv=args.eval_csv,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )


def print_device_info() -> torch.device:
    info = get_device_info()
    device = get_best_device()

    print("\n=== CUDA Readiness ===")
    print(f"CUDA available  : {info.get('cuda_available')}")
    print(f"Selected device : {device}")
    if info.get("gpu_name"):
        print(f"GPU name        : {info.get('gpu_name')}")
    return device


def run_day5_pipeline(config: Day5Config, device: torch.device) -> None:
    print("\n=== Formal Evaluation ===")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(config.eval_csv)
    print(f"Evaluation rows : {len(eval_df)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir)

    eval_dataset = FakeNewsDataset(
        texts=eval_df[config.text_column].astype(str).tolist(),
        labels=eval_df[config.label_column].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    model.to(device)

    prediction_pack = collect_predictions(model, eval_loader, device)
    labels = prediction_pack["labels"]
    preds = prediction_pack["preds"]
    probs = prediction_pack["probs"]

    metrics = compute_metrics(labels, preds)
    confusion = build_confusion_outputs(labels, preds)

    confusion_plot_path = config.output_dir / "confusion_matrix.png"
    save_confusion_matrix_plot(confusion["matrix"], confusion_plot_path)

    threshold_csv_path = config.output_dir / "threshold_analysis.csv"
    pr_curve_path = config.output_dir / "precision_recall_curve.png"
    save_threshold_analysis(labels, probs, threshold_csv_path, pr_curve_path)

    report_path = config.output_dir / "classification_report.txt"
    report_dict = save_classification_report(labels, preds, report_path)

    summary = {
        "model_dir": str(config.model_dir),
        "eval_csv": str(config.eval_csv),
        "rows_evaluated": int(len(eval_df)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "confusion_counts": confusion["counts"],
        "confusion_rates": confusion["rates"],
        "classification_report": report_dict,
        "artifacts": {
            "confusion_matrix_plot": str(confusion_plot_path),
            "precision_recall_curve_plot": str(pr_curve_path),
            "threshold_analysis_csv": str(threshold_csv_path),
            "classification_report_txt": str(report_path),
        },
    }

    summary_path = config.output_dir / "evaluation_summary.json"
    save_summary_json(summary, summary_path)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1_score']:.4f}")

    print("\n=== Confusion Matrix Rates (of total) ===")
    print(f"TN rate   : {confusion['rates']['tn_rate']:.4f}")
    print(f"FP rate   : {confusion['rates']['fp_rate']:.4f}")
    print(f"FN rate   : {confusion['rates']['fn_rate']:.4f}")
    print(f"TP rate   : {confusion['rates']['tp_rate']:.4f}")

    print("\n=== Saved Evaluation Artifacts ===")
    print(f"Summary JSON            : {summary_path}")
    print(f"Classification report   : {report_path}")
    print(f"Confusion matrix plot   : {confusion_plot_path}")
    print(f"Precision-recall curve  : {pr_curve_path}")
    print(f"Threshold analysis CSV  : {threshold_csv_path}")


def main() -> None:
    config = parse_args()
    device = print_device_info()
    run_day5_pipeline(config, device)


if __name__ == "__main__":
    main()
