from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    epoch_seconds: float
    eta_seconds: float
    learning_rate: float
    is_best: bool


def _evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    criterion: Optional[nn.Module] = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        val_bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [val]", leave=False)
        for batch in val_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if criterion is None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_acc = correct / max(total, 1)
            val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    tokenizer=None,
    checkpoint_dir: Optional[Path] = None,
    early_stopping_patience: Optional[int] = None,
    scheduler_patience: Optional[int] = None,
    scheduler_factor: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
) -> List[EpochMetrics]:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = None
    if scheduler_patience is not None and scheduler_patience >= 0:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

    history: List[EpochMetrics] = []
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if class_weights is not None else None
    total_elapsed = 0.0
    best_val_accuracy = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for step, batch in enumerate(train_bar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            if criterion is None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_bar.set_postfix(batch_loss=f"{loss.item():.4f}", avg_loss=f"{(running_train_loss / step):.4f}")

        avg_train_loss = running_train_loss / max(len(train_loader), 1)
        val_loss, val_accuracy = _evaluate(
            model,
            val_loader,
            device,
            epoch=epoch,
            total_epochs=epochs,
            criterion=criterion,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        total_elapsed += epoch_seconds
        avg_epoch_seconds = total_elapsed / epoch
        remaining_epochs = max(epochs - epoch, 0)
        eta_seconds = avg_epoch_seconds * remaining_epochs

        if scheduler is not None:
            scheduler.step(val_loss)

        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(checkpoint_dir)
        else:
            epochs_without_improvement += 1

        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                epoch_seconds=round(epoch_seconds, 2),
                eta_seconds=round(eta_seconds, 2),
                learning_rate=current_lr,
                is_best=is_best,
            )
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"lr={current_lr:.2e} | "
            f"epoch_time={epoch_seconds:.1f}s | "
            f"eta={eta_seconds/60:.1f}m"
        )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best val_acc={best_val_accuracy:.4f}."
            )
            break

    return history
