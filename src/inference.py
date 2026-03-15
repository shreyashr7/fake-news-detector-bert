from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from src.modeling import load_model_and_tokenizer
from src.preprocessing import clean_text


LABEL_MAP: Dict[int, str] = {
    0: "Fake",
    1: "Real",
}


def load_inference_artifacts(model_dir: Path, device: torch.device):
    model, tokenizer = load_model_and_tokenizer(str(model_dir), num_labels=2)
    model.to(device)
    model.eval()
    return model, tokenizer


def _prepare_text(text: str) -> str:
    prepared = clean_text(text)
    return prepared if prepared else ""


def predict_single_text(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 128,
) -> Tuple[int, str]:
    prepared_text = _prepare_text(text)
    pred_idx, label, _ = predict_single_text_with_confidence(
        text=prepared_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )
    return pred_idx, label


def predict_single_text_with_confidence(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 128,
) -> Tuple[int, str, float]:
    prepared_text = _prepare_text(text)
    encoded = tokenizer(
        prepared_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    return pred_idx, LABEL_MAP.get(pred_idx, "Unknown"), confidence


def predict_single_text_with_probabilities(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 128,
) -> Dict[str, float]:
    prepared_text = _prepare_text(text)
    encoded = tokenizer(
        prepared_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    fake_prob = float(probs[0].item())
    real_prob = float(probs[1].item())
    pred_idx = int(torch.argmax(probs).item())

    return {
        "pred_idx": pred_idx,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "confidence": float(max(fake_prob, real_prob)),
    }