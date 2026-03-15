from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class FakeNewsDataset(Dataset):
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def split_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_seed, stratify=labels
    )

    relative_val = val_size / (1.0 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=relative_val, random_state=random_seed, stratify=train_labels
    )

    train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
    val_df = pd.DataFrame({"text": val_texts, "label": val_labels})
    test_df = pd.DataFrame({"text": test_texts, "label": test_labels})

    return train_df, val_df, test_df
