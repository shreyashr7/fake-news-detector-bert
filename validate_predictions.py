from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.device import get_device_info, get_best_device
from src.inference import load_inference_artifacts, predict_single_text_with_probabilities


def run_validation(
    test_csv: Path = Path("artifacts/data_splits/test.csv"),
    model_dir: Path = Path("artifacts/model_final/best_model"),
    output_dir: Path = Path("artifacts/deployment_checks"),
    min_streak: int = 5,
    max_export: int = 10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)
    texts = df["text"].astype(str).fillna("").tolist()
    labels = df["label"].astype(int).tolist()

    info = get_device_info()
    device = get_best_device()
    print("\n=== CUDA Readiness ===")
    print(f"CUDA available  : {info.get('cuda_available')}")
    print(f"Selected device : {device}")
    if info.get('gpu_name'):
        print(f"GPU name        : {info.get('gpu_name')}")
    model, tokenizer = load_inference_artifacts(model_dir, device)

    rows = []
    current_start = 0
    current_len = 0
    first_streak_start: int | None = None
    best_start = 0
    best_len = 0

    for idx, (text, label) in enumerate(zip(texts, labels)):
        prediction = predict_single_text_with_probabilities(text, model, tokenizer, device, max_length=128)
        fake_prob = float(prediction["fake_prob"])
        real_prob = float(prediction["real_prob"])
        pred = 0 if fake_prob >= real_prob else 1
        conf = max(fake_prob, real_prob)
        correct = int(pred == label)

        rows.append(
            {
                "row_index": idx,
                "label": int(label),
                "prediction": int(pred),
                "label_name": "Fake" if label == 0 else "Real",
                "prediction_name": "Fake" if pred == 0 else "Real",
                "confidence": conf,
                "correct": correct,
                "text": " ".join(text.split()),
            }
        )

        if correct:
            if current_len == 0:
                current_start = idx
            current_len += 1
            if first_streak_start is None and current_len >= min_streak:
                first_streak_start = current_start
        else:
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
            current_len = 0

    if current_len > best_len:
        best_len = current_len
        best_start = current_start

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(output_dir / "validation_predictions.csv", index=False)

    if first_streak_start is not None:
        streak_df = rows_df.iloc[first_streak_start : first_streak_start + max(min_streak, max_export)].copy()
        streak_df.to_csv(output_dir / "streak_samples.csv", index=False)
    else:
        streak_df = pd.DataFrame(columns=rows_df.columns)
        streak_df.to_csv(output_dir / "streak_samples.csv", index=False)

    summary = {
        "test_csv": str(test_csv),
        "model_dir": str(model_dir),
        "rows_evaluated": int(len(rows_df)),
        "accuracy": float(rows_df["correct"].mean()) if len(rows_df) else 0.0,
        "min_streak_requested": int(min_streak),
        "first_streak_start_index": None if first_streak_start is None else int(first_streak_start),
        "longest_streak_length": int(best_len),
        "longest_streak_start_index": int(best_start),
        "exports": {
            "all_predictions_csv": str(output_dir / "validation_predictions.csv"),
            "streak_samples_csv": str(output_dir / "streak_samples.csv"),
        },
    }

    with open(output_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_validation()