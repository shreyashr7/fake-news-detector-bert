from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SATIRE_KEYWORDS = [
    "satire",
    "satirical",
    "parody",
    "sarcasm",
    "sarcastic",
    "humor",
    "humour",
    "joke",
    "spoof",
    "ironic",
    "exaggerat",
]

BIASED_LANGUAGE_KEYWORDS = [
    "corrupt",
    "traitor",
    "disgrace",
    "shame",
    "propaganda",
    "rigged",
    "liar",
    "lies",
    "fake",
    "hoax",
    "evil",
    "outrage",
    "agenda",
    "radical",
    "extremist",
]


def build_prediction_frame(
    texts: List[str],
    labels: List[int],
    preds: List[int],
    probs: List[float],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "text": texts,
            "true_label": labels,
            "pred_label": preds,
            "pred_prob_fake": probs,
        }
    )
    frame["error_type"] = "correct"
    frame.loc[(frame["true_label"] == 0) & (frame["pred_label"] == 1), "error_type"] = "false_positive"
    frame.loc[(frame["true_label"] == 1) & (frame["pred_label"] == 0), "error_type"] = "false_negative"
    return frame


def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    lower = str(text).lower()
    return any(keyword in lower for keyword in keywords)


def add_pattern_flags(frame: pd.DataFrame) -> pd.DataFrame:
    analysis = frame.copy()
    analysis["has_satire_signal"] = analysis["text"].map(lambda t: _contains_any_keyword(t, SATIRE_KEYWORDS))
    analysis["has_bias_signal"] = analysis["text"].map(
        lambda t: _contains_any_keyword(t, BIASED_LANGUAGE_KEYWORDS)
    )
    return analysis


def extract_error_samples(
    frame: pd.DataFrame,
    sample_size: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    false_positive = frame[frame["error_type"] == "false_positive"]
    false_negative = frame[frame["error_type"] == "false_negative"]

    fp_sample = false_positive.sample(min(sample_size, len(false_positive)), random_state=random_seed) if len(false_positive) else false_positive
    fn_sample = false_negative.sample(min(sample_size, len(false_negative)), random_state=random_seed) if len(false_negative) else false_negative

    return fp_sample, fn_sample


def compute_error_pattern_stats(frame: pd.DataFrame) -> Dict[str, object]:
    total_rows = len(frame)
    errors = frame[frame["error_type"] != "correct"]
    correct = frame[frame["error_type"] == "correct"]

    def safe_ratio(num: int, den: int) -> float:
        return float(num / den) if den else 0.0

    stats = {
        "rows_total": int(total_rows),
        "rows_correct": int(len(correct)),
        "rows_error": int(len(errors)),
        "false_positive_count": int((frame["error_type"] == "false_positive").sum()),
        "false_negative_count": int((frame["error_type"] == "false_negative").sum()),
        "error_rate": safe_ratio(len(errors), total_rows),
        "satire_signal_rate_in_errors": safe_ratio(int(errors["has_satire_signal"].sum()), len(errors)),
        "bias_signal_rate_in_errors": safe_ratio(int(errors["has_bias_signal"].sum()), len(errors)),
        "satire_signal_rate_in_correct": safe_ratio(int(correct["has_satire_signal"].sum()), len(correct)),
        "bias_signal_rate_in_correct": safe_ratio(int(correct["has_bias_signal"].sum()), len(correct)),
    }

    return stats


def _truncate_text(text: str, limit: int = 240) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def generate_error_report(
    stats: Dict[str, object],
    fp_sample: pd.DataFrame,
    fn_sample: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("Day 6 - Error Analysis Report")
    lines.append("================================")
    lines.append("")
    lines.append("1) Diagnostic Summary")
    lines.append(f"- Total rows analyzed: {stats['rows_total']}")
    lines.append(f"- Correct predictions: {stats['rows_correct']}")
    lines.append(f"- Total errors: {stats['rows_error']}")
    lines.append(f"- False positives: {stats['false_positive_count']}")
    lines.append(f"- False negatives: {stats['false_negative_count']}")
    lines.append(f"- Error rate: {stats['error_rate']:.4f}")
    lines.append("")
    lines.append("2) Pattern Recognition")
    lines.append(
        f"- Satire signal frequency in errors: {stats['satire_signal_rate_in_errors']:.4f} "
        f"(correct baseline: {stats['satire_signal_rate_in_correct']:.4f})"
    )
    lines.append(
        f"- Biased-language signal frequency in errors: {stats['bias_signal_rate_in_errors']:.4f} "
        f"(correct baseline: {stats['bias_signal_rate_in_correct']:.4f})"
    )
    lines.append("- Interpretation: errors concentrate where context is ambiguous, sarcastic, or strongly opinionated.")
    lines.append("")

    lines.append("3) False Positive Samples (Real -> Predicted Fake)")
    if len(fp_sample) == 0:
        lines.append("- No false positives found in this evaluation set.")
    else:
        for idx, row in fp_sample.reset_index(drop=True).iterrows():
            lines.append(
                f"- FP#{idx+1}: prob_fake={row['pred_prob_fake']:.4f} | satire={bool(row['has_satire_signal'])} "
                f"| bias={bool(row['has_bias_signal'])} | text=\"{_truncate_text(row['text'])}\""
            )

    lines.append("")
    lines.append("4) False Negative Samples (Fake -> Predicted Real)")
    if len(fn_sample) == 0:
        lines.append("- No false negatives found in this evaluation set.")
    else:
        for idx, row in fn_sample.reset_index(drop=True).iterrows():
            lines.append(
                f"- FN#{idx+1}: prob_fake={row['pred_prob_fake']:.4f} | satire={bool(row['has_satire_signal'])} "
                f"| bias={bool(row['has_bias_signal'])} | text=\"{_truncate_text(row['text'])}\""
            )

    lines.append("")
    lines.append("5) Improvement Guidance")
    lines.append("- Expand training coverage for satire/parody examples to reduce sarcastic-context confusion.")
    lines.append("- Add biased and opinionated writing samples with balanced labels to improve contextual calibration.")
    lines.append("- Consider threshold tuning using precision-recall trade-offs from Day 5 outputs.")

    output_path.write_text("\n".join(lines), encoding="utf-8")
