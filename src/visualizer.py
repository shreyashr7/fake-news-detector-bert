from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_label_distribution(df: pd.DataFrame, label_col: str, output_dir: Path) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = {0: "Fake", 1: "Real"}
    counts = df[label_col].map(label_map).value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Label Distribution: Fake vs Real News", fontsize=14, fontweight="bold")

    sns.barplot(x=counts.index, y=counts.values, palette=["#e74c3c", "#2ecc71"], ax=axes[0])
    axes[0].set_title("Count per Class")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Label")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.2f%%",
        colors=["#e74c3c", "#2ecc71"],
        startangle=90,
    )
    axes[1].set_title("Class Proportion")

    plot_path = output_dir / "label_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(plot_path)
