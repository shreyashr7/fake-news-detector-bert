from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Day3Config
from src.dataset import FakeNewsDataset
from src.device import get_device_info, get_best_device
from src.modeling import load_model_and_tokenizer
from src.trainer import train_model


def parse_args() -> Day3Config:
    parser = argparse.ArgumentParser(description="Day 3: BERT fine-tuning for binary fake-news classification")
    parser.add_argument("--train-csv", type=Path, default=Path("artifacts/data_splits/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("artifacts/data_splits/val.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_baseline"))
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    return Day3Config(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
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


def run_day3_pipeline(config: Day3Config, device: torch.device) -> None:
    print("\n=== Baseline Training: Model Fine-Tuning ===")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.train_csv)
    val_df = pd.read_csv(config.val_csv)
    print(f"Train rows      : {len(train_df)}")
    print(f"Validation rows : {len(val_df)}")

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

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )

    model_dir = config.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    log_df = pd.DataFrame([h.__dict__ for h in history])
    log_path = config.output_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False)

    final = history[-1] if history else None
    summary = {
        "model_name": config.model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "final_train_loss": final.train_loss if final else None,
        "final_val_loss": final.val_loss if final else None,
        "final_val_accuracy": final.val_accuracy if final else None,
        "saved_model_dir": str(model_dir),
        "training_log_csv": str(log_path),
    }
    summary_path = config.output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Saved Baseline Artifacts ===")
    print(f"Model directory  : {model_dir}")
    print(f"Training log CSV : {log_path}")
    print(f"Summary JSON     : {summary_path}")


def main() -> None:
    config = parse_args()
    device = print_device_info()
    run_day3_pipeline(config, device)


if __name__ == "__main__":
    main()
