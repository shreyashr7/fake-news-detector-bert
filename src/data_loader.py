from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .preprocessing import first_existing

TEXT_CANDIDATES = ["text", "title", "content", "news", "article"]
LABEL_CANDIDATES = ["label", "target", "class", "is_fake"]


def _normalize_binary_labels(series: pd.Series) -> pd.Series:
    def normalize(value):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"fake", "false", "0", "f"}:
                return 0
            if lowered in {"real", "true", "1", "t"}:
                return 1
        if value in {0, 1}:
            return int(value)
        return value

    normalized = series.map(normalize)
    uniques = set(normalized.dropna().unique().tolist())
    if not uniques.issubset({0, 1}):
        raise ValueError(f"Label column is not binary after normalization. Found values: {sorted(uniques)}")
    return normalized.astype(int)


def load_kaggle_dataset(
    single_csv: Optional[Path] = None,
    fake_csv: Optional[Path] = None,
    real_csv: Optional[Path] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> tuple[pd.DataFrame, str, str]:
    if single_csv:
        df = pd.read_csv(single_csv)
        resolved_text_col = text_col or first_existing(TEXT_CANDIDATES, df.columns)
        resolved_label_col = label_col or first_existing(LABEL_CANDIDATES, df.columns)

        if not resolved_text_col or not resolved_label_col:
            raise ValueError(
                "Could not infer text/label columns. Provide --text-column and --label-column explicitly."
            )

        df = df[[resolved_text_col, resolved_label_col]].copy()
        df.rename(columns={resolved_text_col: "text", resolved_label_col: "label"}, inplace=True)
        df["label"] = _normalize_binary_labels(df["label"])
        return df, "text", "label"

    if fake_csv and real_csv:
        fake_df = pd.read_csv(fake_csv)
        real_df = pd.read_csv(real_csv)

        resolved_text_col_fake = text_col or first_existing(TEXT_CANDIDATES, fake_df.columns)
        resolved_text_col_real = text_col or first_existing(TEXT_CANDIDATES, real_df.columns)
        if not resolved_text_col_fake or not resolved_text_col_real:
            raise ValueError("Could not infer text column for Fake/True CSV files. Use --text-column.")

        fake = fake_df[[resolved_text_col_fake]].copy()
        real = real_df[[resolved_text_col_real]].copy()

        fake.rename(columns={resolved_text_col_fake: "text"}, inplace=True)
        real.rename(columns={resolved_text_col_real: "text"}, inplace=True)
        fake["label"] = 0
        real["label"] = 1

        combined = pd.concat([fake, real], ignore_index=True)
        return combined, "text", "label"

    raise ValueError("Provide either --single-csv OR both --fake-csv and --real-csv.")
