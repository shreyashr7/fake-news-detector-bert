# FakeNewsDetector — Full Project Context (New Chat Transfer)

> **Purpose of this file:** Complete context for continuing this project in a new chat session.
> Covers project history, all code decisions, current state, known issues, and next steps.

---

## 1. What This Project Is

Binary fake-news classifier that labels a news headline/article as **Fake (0)** or **Real (1)**.
- Model: `bert-base-uncased` fine-tuned with PyTorch + HuggingFace Transformers
- UI: Streamlit app for inference
- Status: **Fully built and deployed to Streamlit Cloud. Code pushed to GitHub.**

---

## 2. Project Path & Structure

**Local path:** `C:\Users\user\Documents\FakeNewsDetector`

```
FakeNewsDetector/
├── main.py                    # Entry-point (quick run)
├── data_prep.py               # Day 1 – raw data loading & cleaning
├── tokenize_and_split.py      # Day 2 – tokenization & train/val/test split
├── baseline_train.py          # Day 3 – baseline BERT training loop
├── improve_model.py           # Day 4 – improved training (weighted loss, LR scheduler)
├── optimized_train.py         # Day 5 – optimized training (early stopping, tuning)
├── evaluate_model.py          # Day 5 – formal evaluation on val set
├── error_analysis_run.py      # Day 6 – misclassification deep-dive on test set
├── validate_predictions.py    # Day 7 – prediction validation & streak examples
├── streamlit_app.py           # Day 8 – Streamlit inference UI
├── requirements.txt           # pandas, torch, transformers, sklearn, streamlit, etc.
├── README.md
├── context.md                 # ← this file
│
├── src/                       # Shared library modules
│   ├── config.py              # Day1Config…Day8Config dataclasses with all hyperparams/paths
│   ├── dataset.py             # FakeNewsDataset (PyTorch Dataset)
│   ├── data_loader.py         # DataLoader factory
│   ├── device.py              # get_device_info() + get_best_device() with CPU fallback
│   ├── eda.py                 # Exploratory data analysis helpers
│   ├── error_analysis.py      # Error analysis: pattern flags, error report generation
│   ├── evaluator.py           # Metrics: accuracy, F1, confusion matrix, threshold analysis
│   ├── inference.py           # Inference: load artifacts, predict_single_text_with_probabilities
│   ├── modeling.py            # load_model_and_tokenizer (AutoModelForSequenceClassification)
│   ├── preprocessing.py       # clean_text() utility
│   ├── tokenizer_utils.py     # Tokenizer loading helpers
│   ├── trainer.py             # Training loop (loss, AdamW, grad clipping)
│   └── visualizer.py          # Confusion matrix & loss curve plots
│
├── data/                      # Raw and processed CSVs
├── artifacts/                 # Per-phase outputs
│   ├── data_cleaning/         # Was day1
│   ├── data_splits/           # Was day2
│   ├── model_baseline/        # Was day3
│   ├── model_improved/        # Was day4
│   ├── evaluation_optimized/  # Was day5
│   ├── error_analysis/        # Was day6
│   ├── model_final/           # Was day7 (contains final model)
│   ├── deployment_checks/     # Was day8

│
├── .venv/                     # Alternate venv (Python version unknown)
├── .venv311/                  # Active venv – Python 3.11
└── .venv312/                  # Alternate venv – Python 3.12
```

---

## 3. Key Design Decisions

### Label Mapping
| Label | Class |
|-------|-------|
| `0`   | Fake  |
| `1`   | Real  |

**Critical:** This mapping must be consistent everywhere — training, evaluation, inference, Streamlit. Mismatches cause inverted predictions.

### Config System (`src/config.py`)
Each day has a typed dataclass config (Day1Config through Day8Config) with all hyperparameters and artifact paths hardcoded as defaults. Scripts parse CLI args and override defaults.

Key hyperparams settled on:
- `model_name`: `bert-base-uncased`
- `max_length`: 128 (reduced from 512 after Day 3 for speed)
- `batch_size`: 8 (train), 16 (eval)
- `learning_rate`: 2e-5
- `epochs`: 8 with `early_stopping_patience=2`
- `scheduler_patience=1`, `scheduler_factor=0.5`

### Best model location
`artifacts/day7/best_model/` — this is what Streamlit uses (`Day8Config.model_dir`)

### Inference pipeline (`src/inference.py`)
- `load_inference_artifacts(model_dir, device)` → loads model + tokenizer, moves to device, sets eval mode
- `predict_single_text_with_probabilities(text, model, tokenizer, device)` → returns `(label_str, confidence, probabilities_dict)`
- `LABEL_MAP = {0: "Fake", 1: "Real"}`
- Applies `clean_text()` before tokenizing
- Streamlit has short-text guard: warns if input ≤ 4 words

### Device selection (`src/device.py`) — current full code
```python
from __future__ import annotations
from typing import Dict, Any

def get_device_info() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_available": False, "cuda_available": False,
                "selected_device": "cpu", "cuda_version": None,
                "gpu_name": None, "note": "PyTorch not installed; using CPU fallback."}
    cuda_available = bool(torch.cuda.is_available())
    selected_device = "cuda" if cuda_available else "cpu"
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {"torch_available": True, "cuda_available": cuda_available,
            "selected_device": selected_device, "cuda_version": torch.version.cuda,
            "gpu_name": gpu_name, "note": "CUDA available." if cuda_available else "CUDA unavailable."}

def get_best_device():
    import torch
    info = get_device_info()
    if info.get("cuda_available"):
        try:
            t = torch.tensor([1.0], device="cuda")
            t += 1
            return torch.device("cuda")
        except Exception:
            print("CUDA detected but kernel image unsupported; falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")
```

All pipeline scripts (baseline_train, improve_model, optimized_train, evaluate_model, error_analysis_run, validate_predictions, streamlit_app) import and use both `get_device_info` and `get_best_device` from `src.device`.

---

## 4. Environment Setup

### Active environment: `.venv311` (Python 3.11)

```powershell
# Activate
.\.venv311\Scripts\activate

# Install deps
pip install -r requirements.txt
```

### Installed torch build
`torch==2.12.0.dev20260315+cu128` (Nightly build)

### Other venvs present
| Folder    | Python | torch version | Notes                     |
|-----------|--------|---------------|---------------------------|
| `.venv311`| 3.11   | 2.12.0.dev..  | **Active — use this**     |
| `.venv312`| 3.12   | unknown       | Not verified              |
| `.venv`   | ?      | unknown       | Not verified              |

---

## 5. GPU Issue — Resolved ✅

### Hardware
- **GPU:** NVIDIA GeForce RTX 5060 Ti
- **Compute capability:** sm_120 (Blackwell architecture, Gen 12)

### Resolution
Installed PyTorch Nightly build (cu128) which supports `sm_120` kernels.

**Status:** `get_best_device()` correctly returns `cuda` and tensors operations work.

### Previous Issue
PyTorch 2.5.1+cu121 did not include kernels for sm_120, causing RuntimeError on tensor ops despite `is_available()` returning True. The project was running on CPU fallback.


---

## 6. Running the Pipeline

```powershell
.\.venv311\Scripts\activate

# Full pipeline (run in order)
python data_prep.py
python tokenize_and_split.py
python baseline_train.py
python improve_model.py
python optimized_train.py
python evaluate_model.py
python error_analysis_run.py
python validate_predictions.py

# Streamlit UI
streamlit run streamlit_app.py
```

All scripts accept CLI overrides. Example:
```powershell
python evaluate_model.py --model-dir artifacts/day7/best_model --eval-csv artifacts/day2/val.csv
```

---

## 7. Artifacts Phase Mapping

| Phase | Old Folder | New Folder | Contents |
|-------|------------|------------|----------|
| Data Prep | `artifacts/day1/` | `artifacts/data_cleaning/` | `cleaned_dataset.csv` |
| Splits | `artifacts/day2/` | `artifacts/data_splits/` | `train.csv`, `val.csv`, `test.csv` |
| Baseline | `artifacts/day3/` | `artifacts/model_baseline/` | Baseline checkpoint |
| Improved | `artifacts/day4/` | `artifacts/model_improved/` | Best model (weighted loss) |
| Eval | `artifacts/day5/` | `artifacts/evaluation_optimized/` | Metrics, `evaluation_summary.json` |
| Errors | `artifacts/day6/` | `artifacts/error_analysis/` | `error_analysis_report.txt` |
| Final | `artifacts/day7/` | `artifacts/model_final/` | **Final model** used by app |
| Deploy | `artifacts/day8/` | `artifacts/deployment_checks/` | `streak_samples.csv`, `validation_summary.json` |

---

## 8. What's Done vs Pending

### Done ✅
- Full 8-day pipeline implemented and run
- `bert-base-uncased` fine-tuned, best model saved to `artifacts/day7/best_model/`
- Streamlit app built and deployed to Streamlit Cloud
- Code pushed to GitHub
- **GPU inference works** (PyTorch Nightly sm_120 support)
- All scripts use shared `src/` modules for consistency
- README with run instructions

### Pending / Known Issues
- `.venv` and `.venv312` torch versions not verified — may or may not have GPU support
- No unit tests

---

## 9. requirements.txt

```
pandas>=2.0.0
torch>=2.2.0
transformers>=4.35.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
streamlit>=1.28.0
```

---

## 10. Quick Reference

| Task                        | Command                                                     |
|-----------------------------|-------------------------------------------------------------|
| Activate env                | `.\.venv311\Scripts\activate`                               |
| Run full pipeline           | `python data_prep.py` → ... → `python validate_predictions.py` |
| Launch Streamlit locally    | `streamlit run streamlit_app.py`                            |
| Check GPU                   | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` |
| Test GPU compute            | `python -c "import torch; t=torch.tensor([1.0],device='cuda'); print(t+1)"` |
| Check current torch version | `python -c "import torch; print(torch.__version__)"` |
