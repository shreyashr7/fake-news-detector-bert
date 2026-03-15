from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Day1Config:
    output_dir: Path = Path("artifacts/data_cleaning")
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    fake_csv: Optional[Path] = None
    real_csv: Optional[Path] = None
    single_csv: Optional[Path] = None
    random_seed: int = 42
    sample_count: int = 5


@dataclass
class Day2Config:
    cleaned_csv: Path = Path("artifacts/data_cleaning/cleaned_dataset.csv")
    output_dir: Path = Path("artifacts/data_splits")
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    text_column: str = "clean_text"
    label_column: str = "label"


@dataclass
class Day3Config:
    train_csv: Path = Path("artifacts/data_splits/train.csv")
    val_csv: Path = Path("artifacts/data_splits/val.csv")
    output_dir: Path = Path("artifacts/model_baseline")
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3
    random_seed: int = 42
    text_column: str = "text"
    label_column: str = "label"


@dataclass
class Day4Config:
    train_csv: Path = Path("artifacts/data_splits/train.csv")
    val_csv: Path = Path("artifacts/data_splits/val.csv")
    output_dir: Path = Path("artifacts/model_improved")
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 8
    early_stopping_patience: int = 2
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5
    random_seed: int = 42
    text_column: str = "text"
    label_column: str = "label"


@dataclass
class Day5Config:
    eval_csv: Path = Path("artifacts/data_splits/val.csv")
    model_dir: Path = Path("artifacts/model_improved/best_model")
    output_dir: Path = Path("artifacts/evaluation_optimized")
    max_length: int = 128
    batch_size: int = 16
    text_column: str = "text"
    label_column: str = "label"


@dataclass
class Day6Config:
    test_csv: Path = Path("artifacts/data_splits/test.csv")
    model_dir: Path = Path("artifacts/model_improved/best_model")
    output_dir: Path = Path("artifacts/error_analysis")
    max_length: int = 128
    batch_size: int = 16
    sample_size: int = 10
    random_seed: int = 42
    text_column: str = "text"
    label_column: str = "label"


@dataclass
class Day7Config:
    train_csv: Path = Path("artifacts/data_splits/train.csv")
    val_csv: Path = Path("artifacts/data_splits/val.csv")
    baseline_summary_json: Path = Path("artifacts/evaluation_optimized/evaluation_summary.json")
    output_dir: Path = Path("artifacts/model_final")
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 8
    early_stopping_patience: int = 2
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5
    random_seed: int = 42
    text_column: str = "text"
    label_column: str = "label"


@dataclass
class Day8Config:
    model_dir: Path = Path("artifacts/model_final/best_model")
    max_length: int = 128
    app_title: str = "Fake News Detector"
    app_description: str = "Enter a headline or article to classify it as Real or Fake."
    text_area_label: str = "News text"
    button_label: str = "Predict"
