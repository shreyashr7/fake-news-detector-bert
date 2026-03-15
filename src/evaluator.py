from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def collect_predictions(model, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        bar = tqdm(loader, desc="Day 5 [eval]", leave=False)
        for batch in bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return {
        "labels": np.array(all_labels, dtype=np.int64),
        "preds": np.array(all_preds, dtype=np.int64),
        "probs": np.array(all_probs, dtype=np.float32),
    }


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
    }


def build_confusion_outputs(labels: np.ndarray, preds: np.ndarray) -> Dict[str, object]:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = max(int(cm.sum()), 1)

    rates = {
        "tn_rate": float(tn / total),
        "fp_rate": float(fp / total),
        "fn_rate": float(fn / total),
        "tp_rate": float(tp / total),
    }

    return {
        "matrix": cm,
        "counts": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "rates": rates,
    }


def save_confusion_matrix_plot(cm: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Fake (0)", "Pred Real (1)"],
        yticklabels=["True Fake (0)", "True Real (1)"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_threshold_analysis(labels: np.ndarray, probs: np.ndarray, output_csv: Path, curve_plot_path: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    curve_plot_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, thresholds = precision_recall_curve(labels, probs)

    rows = []
    for idx, threshold in enumerate(thresholds):
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision[idx + 1]),
                "recall": float(recall[idx + 1]),
                "f1_score": float(
                    0.0
                    if (precision[idx + 1] + recall[idx + 1]) == 0
                    else (2 * precision[idx + 1] * recall[idx + 1]) / (precision[idx + 1] + recall[idx + 1])
                ),
            }
        )

    import pandas as pd

    pd.DataFrame(rows).to_csv(output_csv, index=False)

    plt.figure(figsize=(6, 5))
    plt.plot(recall[:-1], precision[:-1], linewidth=2)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_plot_path, dpi=150)
    plt.close()


def save_classification_report(labels: np.ndarray, preds: np.ndarray, output_path: Path) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_dict = classification_report(
        labels,
        preds,
        target_names=["Fake", "Real"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        labels,
        preds,
        target_names=["Fake", "Real"],
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_dict


def save_summary_json(summary: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
