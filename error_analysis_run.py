from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import Day6Config
from src.dataset import FakeNewsDataset
from src.device import get_device_info, get_best_device
from src.error_analysis import (
    add_pattern_flags,
    build_prediction_frame,
    compute_error_pattern_stats,
    extract_error_samples,
    generate_error_report,
)


def parse_args() -> Day6Config:
    parser = argparse.ArgumentParser(description="Day 6: Error analysis on model misclassifications")
    parser.add_argument("--test-csv", type=Path, default=Path("artifacts/data_splits/test.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model_improved/best_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/error_analysis"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    return Day6Config(
        test_csv=args.test_csv,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
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


def run_day6_pipeline(config: Day6Config, device: torch.device) -> None:
    print("\n=== Error Analysis ===")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(config.test_csv)
    print(f"Test rows       : {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir)
    model.to(device)
    model.eval()

    texts = test_df[config.text_column].astype(str).tolist()
    labels = test_df[config.label_column].astype(int).tolist()

    dataset = FakeNewsDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    pred_frame = build_prediction_frame(texts, labels, all_preds, all_probs)
    analysis_frame = add_pattern_flags(pred_frame)

    fp_sample, fn_sample = extract_error_samples(
        analysis_frame,
        sample_size=config.sample_size,
        random_seed=config.random_seed,
    )
    stats = compute_error_pattern_stats(analysis_frame)

    full_results_path = config.output_dir / "test_predictions_with_errors.csv"
    fp_path = config.output_dir / "false_positives_sample.csv"
    fn_path = config.output_dir / "false_negatives_sample.csv"
    report_path = config.output_dir / "error_analysis_report.txt"
    summary_path = config.output_dir / "error_analysis_summary.json"

    analysis_frame.to_csv(full_results_path, index=False)
    fp_sample.to_csv(fp_path, index=False)
    fn_sample.to_csv(fn_path, index=False)
    generate_error_report(stats, fp_sample, fn_sample, report_path)

    summary = {
        "model_dir": str(config.model_dir),
        "test_csv": str(config.test_csv),
        "rows_total": stats["rows_total"],
        "rows_correct": stats["rows_correct"],
        "rows_error": stats["rows_error"],
        "error_rate": stats["error_rate"],
        "false_positive_count": stats["false_positive_count"],
        "false_negative_count": stats["false_negative_count"],
        "satire_signal_rate_in_errors": stats["satire_signal_rate_in_errors"],
        "bias_signal_rate_in_errors": stats["bias_signal_rate_in_errors"],
        "satire_signal_rate_in_correct": stats["satire_signal_rate_in_correct"],
        "bias_signal_rate_in_correct": stats["bias_signal_rate_in_correct"],
        "artifacts": {
            "full_results_csv": str(full_results_path),
            "false_positives_sample_csv": str(fp_path),
            "false_negatives_sample_csv": str(fn_path),
            "error_analysis_report": str(report_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Analysis Summary ===")
    print(f"Total errors     : {stats['rows_error']} / {stats['rows_total']} ({stats['error_rate']:.4%})")
    print(f"False positives  : {stats['false_positive_count']}")
    print(f"False negatives  : {stats['false_negative_count']}")
    print(f"Satire signals in errors : {stats['satire_signal_rate_in_errors']:.4f}")
    print(f"Bias signals in errors   : {stats['bias_signal_rate_in_errors']:.4f}")

    print("\n=== Saved Analysis Artifacts ===")
    print(f"Full predictions CSV : {full_results_path}")
    print(f"FP sample CSV        : {fp_path}")
    print(f"FN sample CSV        : {fn_path}")
    print(f"Error report         : {report_path}")
    print(f"Summary JSON         : {summary_path}")


def main() -> None:
    config = parse_args()
    device = print_device_info()
    run_day6_pipeline(config, device)


if __name__ == "__main__":
    main()
