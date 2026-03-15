from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = HTML_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized.lower()


def preprocess_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in dataframe.")

    processed = df.copy()
    processed["clean_text"] = processed[text_col].fillna("").map(clean_text)
    processed = processed[processed["clean_text"].str.len() > 0].reset_index(drop=True)
    return processed


def deduplicate_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    deduped = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    return deduped


def first_existing(candidates: Iterable[str], columns: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None
