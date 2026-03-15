# Fake News Detector

## Problem Statement
Build a binary fake-news detector that classifies a news headline/article sentence as **Fake** or **Real** using BERT. The model should be trained on labeled data and deployed using Streamlit for inference.

## Approach
1. Data preprocessing and tokenization.
2. Fine-tune a pretrained `bert-base-uncased` model using a custom training loop.
3. Evaluate model on validation and test splits with classification metrics and confusion matrix.
4. Analyze misclassifications in error analysis.
5. Deploy with Streamlit and inference helper functions.

## Model Used
- Transformer: `bert-base-uncased`
- Framework: PyTorch + HuggingFace Transformers
- Binary classification head on top of BERT outputs

## Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Validation and test accuracy from final model runs

## Improvements
- Added weighted loss training to handle class imbalance.
- Added validation scripts to confirm predictions and save streak examples.
- Added robust inference utility with confidence and uncertainty handling.
- Added short-input guard in Streamlit for very short or low-context text.

## Key Learnings
- Pretrained transformers can be fine-tuned for domain-specific text classification.
- Correct label mapping (0=Fake, 1=Real) is critical for deploying reliable inference.
- Model confidence thresholds help show uncertain predictions instead of overconfident wrong labels.
- Packaging with clear script names and modular design (`src/`) improves reproducibility.

## Run locally
1. Create environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Train and evaluate:
   ```bash
   python improve_model.py
   python validate_predictions.py
   ```
3. Run Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Optional commands
### 1) Quick device check
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'cuda_version:', torch.version.cuda)"
```

### 2) Full pipeline (from raw data)
```bash
python data_prep.py
python tokenize_and_split.py
python baseline_train.py
python improve_model.py
python optimized_train.py
python evaluate_model.py
python error_analysis_run.py
python validate_predictions.py
```

### 3) Evaluate-only (no retraining)
```bash
python evaluate_model.py
python validate_predictions.py
```

## GPU setup (optional)
If you want GPU acceleration, install CUDA-compatible PyTorch (example for CUDA 13.1):
```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
Then verify:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
