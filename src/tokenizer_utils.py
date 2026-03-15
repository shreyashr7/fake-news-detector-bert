from __future__ import annotations

from typing import Dict, List

from transformers import AutoTokenizer


def load_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_sample(tokenizer: AutoTokenizer, text: str, max_length: int = 512) -> Dict:
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def tokenize_batch(tokenizer: AutoTokenizer, texts: List[str], max_length: int = 512) -> Dict:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def describe_tokenized_output(encoded: Dict) -> Dict:
    return {
        key: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for key, tensor in encoded.items()
    }
