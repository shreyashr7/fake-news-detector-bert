from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd


def compute_dataset_summary(df: pd.DataFrame, text_col: str, label_col: str) -> Dict[str, Any]:
    label_counts = df[label_col].value_counts().sort_index()
    total = int(len(df))
    imbalance_ratio = None
    if len(label_counts) == 2 and label_counts.min() > 0:
        imbalance_ratio = float(label_counts.max() / label_counts.min())

    text_lengths = df[text_col].astype(str).str.len()
    return {
        "num_rows": total,
        "num_columns": int(df.shape[1]),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "label_percentages": {str(k): round((v / total) * 100, 2) for k, v in label_counts.items()},
        "imbalance_ratio_major_to_minor": round(imbalance_ratio, 4) if imbalance_ratio else None,
        "avg_text_length": round(float(text_lengths.mean()), 2) if total else 0.0,
        "median_text_length": float(text_lengths.median()) if total else 0.0,
        "min_text_length": int(text_lengths.min()) if total else 0,
        "max_text_length": int(text_lengths.max()) if total else 0,
    }


def export_day1_artifacts(df: pd.DataFrame, text_col: str, label_col: str, output_dir: Path, sample_count: int = 5) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_dataset_summary(df, text_col=text_col, label_col=label_col)

    summary_path = output_dir / "dataset_summary.csv"
    summary_df = pd.DataFrame(
        [{"metric": key, "value": str(value)} for key, value in summary.items() if key not in {"label_counts", "label_percentages"}]
    )
    summary_df.to_csv(summary_path, index=False)

    label_dist_path = output_dir / "label_distribution.csv"
    label_dist_df = (
        df[label_col]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )
    label_dist_df["percentage"] = (label_dist_df["count"] / len(df) * 100).round(2)
    label_dist_df.to_csv(label_dist_path, index=False)

    samples_path = output_dir / "sample_entries.csv"
    samples = df[[text_col, "clean_text", label_col]].sample(n=min(sample_count, len(df)), random_state=42)
    samples.to_csv(samples_path, index=False)

    cleaned_path = output_dir / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)

    return {
        "summary_path": str(summary_path),
        "label_distribution_path": str(label_dist_path),
        "samples_path": str(samples_path),
        "cleaned_dataset_path": str(cleaned_path),
        "summary": summary,
    }
