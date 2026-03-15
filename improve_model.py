from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import Day7Config
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
from src.modeling import load_model_and_tokenizer
from src.trainer import train_model


def parse_args() -> Day7Config:
    parser = argparse.ArgumentParser(description="Day 7: Model improvement using class-weighted loss")
    parser.add_argument("--train-csv", type=Path, default=Path("artifacts/data_splits/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("artifacts/data_splits/val.csv"))
    parser.add_argument("--baseline-summary-json", type=Path, default=Path("artifacts/evaluation_optimized/evaluation_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_final"))
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--scheduler-patience", type=int, default=1)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    args = parser.parse_args()

    return Day7Config(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        baseline_summary_json=args.baseline_summary_json,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
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


def compute_inverse_frequency_weights(labels: pd.Series) -> torch.Tensor:
    counts = labels.value_counts().sort_index()
    total = float(counts.sum())
    num_classes = int(len(counts))

    weights = []
    for class_idx in range(num_classes):
        class_count = float(counts.get(class_idx, 0.0))
        weight = 0.0 if class_count == 0 else total / (num_classes * class_count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def run_day7_pipeline(config: Day7Config, device: torch.device) -> None:
    print("\n=== Model Improvement (Weighted Loss) ===")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.train_csv)
    val_df = pd.read_csv(config.val_csv)
    print(f"Train rows      : {len(train_df)}")
    print(f"Validation rows : {len(val_df)}")

    class_weights = compute_inverse_frequency_weights(train_df[config.label_column].astype(int))
    print(f"Class weights   : {class_weights.tolist()}")

    model, tokenizer = load_model_and_tokenizer(config.model_name, num_labels=2)

    train_dataset = FakeNewsDataset(
        texts=train_df[config.text_column].astype(str).tolist(),
        labels=train_df[config.label_column].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    val_dataset = FakeNewsDataset(
        texts=val_df[config.text_column].astype(str).tolist(),
        labels=val_df[config.label_column].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    best_checkpoint_dir = config.output_dir / "best_model"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        tokenizer=tokenizer,
        checkpoint_dir=best_checkpoint_dir,
        early_stopping_patience=config.early_stopping_patience,
        scheduler_patience=config.scheduler_patience,
        scheduler_factor=config.scheduler_factor,
        class_weights=class_weights,
    )

    final_model_dir = config.output_dir / "model_final"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    train_log_path = config.output_dir / "training_log.csv"
    pd.DataFrame([h.__dict__ for h in history]).to_csv(train_log_path, index=False)

    best_row = max(history, key=lambda x: x.val_accuracy) if history else None
    final_row = history[-1] if history else None

    train_summary = {
        "model_name": config.model_name,
        "train_csv": str(config.train_csv),
        "val_csv": str(config.val_csv),
        "epochs_requested": config.epochs,
        "epochs_completed": len(history),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "early_stopping_patience": config.early_stopping_patience,
        "scheduler_patience": config.scheduler_patience,
        "scheduler_factor": config.scheduler_factor,
        "class_weights": [float(x) for x in class_weights.tolist()],
        "final_train_loss": final_row.train_loss if final_row else None,
        "final_val_loss": final_row.val_loss if final_row else None,
        "final_val_accuracy": final_row.val_accuracy if final_row else None,
        "best_epoch": best_row.epoch if best_row else None,
        "best_val_accuracy": best_row.val_accuracy if best_row else None,
        "best_val_loss": best_row.val_loss if best_row else None,
        "best_model_dir": str(best_checkpoint_dir),
        "final_model_dir": str(final_model_dir),
        "training_log_csv": str(train_log_path),
    }
    train_summary_path = config.output_dir / "training_summary.json"
    save_summary_json(train_summary, train_summary_path)

    eval_output_dir = config.output_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_dir)
    eval_tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_dir)
    eval_model.to(device)

    eval_dataset = FakeNewsDataset(
        texts=val_df[config.text_column].astype(str).tolist(),
        labels=val_df[config.label_column].astype(int).tolist(),
        tokenizer=eval_tokenizer,
        max_length=config.max_length,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    prediction_pack = collect_predictions(eval_model, eval_loader, device)
    labels = prediction_pack["labels"]
    preds = prediction_pack["preds"]
    probs = prediction_pack["probs"]

    metrics = compute_metrics(labels, preds)
    confusion = build_confusion_outputs(labels, preds)

    confusion_plot_path = eval_output_dir / "confusion_matrix.png"
    threshold_csv_path = eval_output_dir / "threshold_analysis.csv"
    pr_curve_path = eval_output_dir / "precision_recall_curve.png"
    report_path = eval_output_dir / "classification_report.txt"

    save_confusion_matrix_plot(confusion["matrix"], confusion_plot_path)
    save_threshold_analysis(labels, probs, threshold_csv_path, pr_curve_path)
    report_dict = save_classification_report(labels, preds, report_path)

    eval_summary = {
        "model_dir": str(best_checkpoint_dir),
        "eval_csv": str(config.val_csv),
        "rows_evaluated": int(len(val_df)),
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
    eval_summary_path = eval_output_dir / "eval_summary.json"
    save_summary_json(eval_summary, eval_summary_path)

    baseline = {}
    if config.baseline_summary_json.exists():
        with open(config.baseline_summary_json, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    comparison = {
        "baseline_summary_json": str(config.baseline_summary_json),
        "improved_summary_json": str(eval_summary_path),
        "baseline_metrics": {
            "accuracy": baseline.get("accuracy"),
            "precision": baseline.get("precision"),
            "recall": baseline.get("recall"),
            "f1_score": baseline.get("f1_score"),
        },
        "improved_metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        },
        "delta_metrics": {
            "accuracy": None if baseline.get("accuracy") is None else float(metrics["accuracy"] - baseline["accuracy"]),
            "precision": None if baseline.get("precision") is None else float(metrics["precision"] - baseline["precision"]),
            "recall": None if baseline.get("recall") is None else float(metrics["recall"] - baseline["recall"]),
            "f1_score": None if baseline.get("f1_score") is None else float(metrics["f1_score"] - baseline["f1_score"]),
        },
    }
    comparison_path = config.output_dir / "baseline_comparison.json"
    save_summary_json(comparison, comparison_path)

    print("\n=== Improved Metrics ===")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1_score']:.4f}")

    print("\n=== Saved Artifacts ===")
    print(f"Training summary      : {train_summary_path}")
    print(f"Evaluation summary    : {eval_summary_path}")
    print(f"Baseline comparison   : {comparison_path}")
    print(f"Classification report : {report_path}")


if __name__ == "__main__":
    config = parse_args()
    device = print_device_info()
    run_day7_pipeline(config, device)
