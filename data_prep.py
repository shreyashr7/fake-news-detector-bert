from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Day1Config
from src.data_loader import load_kaggle_dataset
from src.device import get_device_info
from src.eda import export_day1_artifacts
from src.preprocessing import deduplicate_text, preprocess_dataframe


def parse_args() -> Day1Config:
	parser = argparse.ArgumentParser(
		description="Day 1 pipeline: Fake News dataset ingestion, EDA, cleaning, and artifact export."
	)
	parser.add_argument("--single-csv", type=Path, default=None, help="Path to one CSV containing both text and labels.")
	parser.add_argument("--fake-csv", type=Path, default=None, help="Path to CSV containing fake news samples.")
	parser.add_argument("--real-csv", type=Path, default=None, help="Path to CSV containing real news samples.")
	parser.add_argument("--text-column", type=str, default=None, help="Optional text column override.")
	parser.add_argument("--label-column", type=str, default=None, help="Optional label column override.")
	parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_cleaning"), help="Directory for Day 1 deliverables.")
	parser.add_argument("--sample-count", type=int, default=5, help="Number of sample entries to export.")

	args = parser.parse_args()
	return Day1Config(
		output_dir=args.output_dir,
		text_column=args.text_column,
		label_column=args.label_column,
		fake_csv=args.fake_csv,
		real_csv=args.real_csv,
		single_csv=args.single_csv,
		sample_count=args.sample_count,
	)


def print_device_info() -> None:
	device_info = get_device_info()
	print("\n=== CUDA Readiness ===")
	print(f"Torch available : {device_info.get('torch_available')}")
	print(f"CUDA available  : {device_info.get('cuda_available')}")
	print(f"Selected device : {device_info.get('selected_device')}")
	print(f"CUDA version    : {device_info.get('cuda_version')}")
	if device_info.get("gpu_name"):
		print(f"GPU name        : {device_info.get('gpu_name')}")
	if device_info.get("note"):
		print(f"Note            : {device_info.get('note')}")


def run_day1_pipeline(config: Day1Config) -> None:
	print("\n=== Dataset Acquisition + EDA + Cleaning ===")

	df, text_col, label_col = load_kaggle_dataset(
		single_csv=config.single_csv,
		fake_csv=config.fake_csv,
		real_csv=config.real_csv,
		text_col=config.text_column,
		label_col=config.label_column,
	)

	before_dropna = len(df)
	df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
	after_dropna = len(df)

	before_dedup = len(df)
	df = deduplicate_text(df, text_col=text_col)
	after_dedup = len(df)

	processed_df = preprocess_dataframe(df, text_col=text_col)

	artifacts = export_day1_artifacts(
		processed_df,
		text_col=text_col,
		label_col=label_col,
		output_dir=config.output_dir,
		sample_count=config.sample_count,
	)

	print("\n=== Data Integrity Checks ===")
	print(f"Rows before null-drop     : {before_dropna}")
	print(f"Rows after null-drop      : {after_dropna}")
	print(f"Rows before dedup         : {before_dedup}")
	print(f"Rows after dedup          : {after_dedup}")
	print(f"Rows after cleaning       : {len(processed_df)}")

	summary = artifacts["summary"]
	print("\n=== Dataset Summary ===")
	print(f"Rows                      : {summary['num_rows']}")
	print(f"Columns                   : {summary['num_columns']}")
	print(f"Label counts              : {summary['label_counts']}")
	print(f"Label percentages         : {summary['label_percentages']}")
	print(f"Imbalance ratio (major/minor): {summary['imbalance_ratio_major_to_minor']}")
	print(f"Avg text length           : {summary['avg_text_length']}")

	print(f"\n=== Saved Data Prep Artifacts ===")
	print(f"Summary CSV               : {artifacts['summary_path']}")
	print(f"Label distribution CSV    : {artifacts['label_distribution_path']}")
	print(f"Sample entries CSV        : {artifacts['samples_path']}")
	print(f"Cleaned dataset CSV       : {artifacts['cleaned_dataset_path']}")


def main() -> None:
	config = parse_args()
	print_device_info()
	run_day1_pipeline(config)


if __name__ == "__main__":
	main()
