from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import Day2Config
from src.dataset import FakeNewsDataset, split_dataset
from src.device import get_device_info
from src.tokenizer_utils import describe_tokenized_output, load_tokenizer, tokenize_sample
from src.visualizer import plot_label_distribution


def parse_args() -> Day2Config:
    parser = argparse.ArgumentParser(
        description="Day 2 pipeline: EDA visualization, tokenization, and train/val/test split."
    )
    parser.add_argument("--cleaned-csv", type=Path, default=Path("artifacts/data_cleaning/cleaned_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_splits"))
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--text-column", type=str, default="clean_text")
    parser.add_argument("--label-column", type=str, default="label")
    args = parser.parse_args()
    return Day2Config(
        cleaned_csv=args.cleaned_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        test_size=args.test_size,
        val_size=args.val_size,
        text_column=args.text_column,
        label_column=args.label_column,
    )


def print_device_info() -> None:
    info = get_device_info()
    print("\n=== CUDA Readiness ===")
    print(f"CUDA available  : {info.get('cuda_available')}")
    print(f"Selected device : {info.get('selected_device')}")
    if info.get("gpu_name"):
        print(f"GPU name        : {info.get('gpu_name')}")


def run_day2_pipeline(config: Day2Config) -> None:
    print("\n=== Processing: Visualization + Tokenization + Split ===")

    df = pd.read_csv(config.cleaned_csv)
    print(f"Loaded cleaned dataset : {len(df)} rows")

    # --- EDA: Label Distribution Visualization ---
    print("\n--- Label Distribution Visualization ---")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_label_distribution(df, label_col=config.label_column, output_dir=config.output_dir)
    print(f"Plot saved             : {plot_path}")

    # --- Tokenizer ---
    print(f"\n--- Loading Tokenizer: {config.model_name} ---")
    tokenizer = load_tokenizer(config.model_name)
    print(f"Vocab size             : {tokenizer.vocab_size}")
    print(f"Model max length       : {tokenizer.model_max_length}")
    print(f"Applied max length     : {config.max_length}")

    # --- Example tokenized output ---
    sample_text = str(df[config.text_column].iloc[0])
    print(f"\nSample text (first 120 chars): {sample_text[:120]}...")
    encoded_sample = tokenize_sample(tokenizer, sample_text, max_length=config.max_length)
    tensor_info = describe_tokenized_output(encoded_sample)

    print("\n--- Tokenized Output Tensor Shapes ---")
    for key, info in tensor_info.items():
        print(f"  {key:20s} shape={info['shape']}  dtype={info['dtype']}")

    first_tokens = tokenizer.convert_ids_to_tokens(encoded_sample["input_ids"][0][:20].tolist())
    print(f"\nFirst 20 tokens        : {first_tokens}")

    # --- Train / Val / Test Split ---
    print("\n--- Dataset Partitioning (Stratified) ---")
    train_df, val_df, test_df = split_dataset(
        df,
        text_col=config.text_column,
        label_col=config.label_column,
        test_size=config.test_size,
        val_size=config.val_size,
        random_seed=config.random_seed,
    )
    print(f"Train size             : {len(train_df)}")
    print(f"Val size               : {len(val_df)}")
    print(f"Test size              : {len(test_df)}")

    train_df.to_csv(config.output_dir / "train.csv", index=False)
    val_df.to_csv(config.output_dir / "val.csv", index=False)
    test_df.to_csv(config.output_dir / "test.csv", index=False)

    # --- PyTorch Dataset shape check ---
    print("\n--- PyTorch Dataset Tensor Shape Check (3 samples) ---")
    sample_ds = FakeNewsDataset(
        texts=train_df["text"].tolist()[:3],
        labels=train_df["label"].tolist()[:3],
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    sample_item = sample_ds[0]
    for key, tensor in sample_item.items():
        print(f"  {key:20s} shape={list(tensor.shape)}  dtype={tensor.dtype}")

    # --- Save artifact summary ---
    artifact_summary = {
        "label_distribution_plot": plot_path,
        "tokenizer_model": config.model_name,
        "max_length": config.max_length,
        "vocab_size": tokenizer.vocab_size,
        "tensor_shapes": tensor_info,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "splits_saved": {
            "train": str(config.output_dir / "train.csv"),
            "val": str(config.output_dir / "val.csv"),
            "test": str(config.output_dir / "test.csv"),
        },
    }
    summary_path = config.output_dir / "splits_summary.json"
    with open(summary_path, "w") as f:
        json.dump(artifact_summary, f, indent=2)

    print("\n=== Saved Processing Artifacts ===")
    print(f"Visualization          : {plot_path}")
    print(f"Train split            : {config.output_dir / 'train.csv'}")
    print(f"Val split              : {config.output_dir / 'val.csv'}")
    print(f"Test split             : {config.output_dir / 'test.csv'}")
    print(f"Summary JSON           : {summary_path}")


def main() -> None:
    config = parse_args()
    print_device_info()
    run_day2_pipeline(config)


if __name__ == "__main__":
    main()
