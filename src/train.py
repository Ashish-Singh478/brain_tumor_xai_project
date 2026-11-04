"""Training and evaluation entry point for the brain-tumor classifiers."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from .dataset import CLASS_NAMES, DataConfig, build_dataloaders
from .model import MODEL_NAMES, create_model


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str = "resnet50"
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 25
    dropout: float = 0.4
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(
            precision_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
    }


def train(
    data_root: str | Path,
    output_dir: str | Path,
    model_config: TrainingConfig,
    image_size: int = 200,
    device: str | None = None,
) -> dict[str, list[dict[str, float]]]:
    seed_everything(model_config.seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders = build_dataloaders(
        DataConfig(
            root=Path(data_root),
            image_size=image_size,
            batch_size=model_config.batch_size,
            seed=model_config.seed,
        )
    )
    model = create_model(
        architecture=model_config.architecture,
        dropout=model_config.dropout,
        pretrained=True,
    ).to(selected_device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=model_config.learning_rate)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[dict[str, float]]] = {"train": [], "validation": []}
    best_validation_loss = float("inf")
    for epoch in range(1, model_config.epochs + 1):
        train_metrics = _run_epoch(
            model, loaders["train"], loss_fn, selected_device, optimizer
        )
        validation_metrics = _run_epoch(
            model, loaders["validation"], loss_fn, selected_device, None
        )
        history["train"].append(train_metrics)
        history["validation"].append(validation_metrics)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            torch.save(
                {
                    "architecture": model_config.architecture,
                    "class_names": CLASS_NAMES,
                    "model_state": model.state_dict(),
                    "training_config": asdict(model_config),
                    "image_size": image_size,
                },
                output_path / "best_model.pt",
            )
        print(
            f"Epoch {epoch:02d}/{model_config.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f}"
        )

    test_metrics = _run_epoch(model, loaders["test"], loss_fn, selected_device, None)
    history["test"] = [test_metrics]
    (output_path / "history.json").write_text(json.dumps(history, indent=2))
    return history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--architecture", choices=sorted(MODEL_NAMES), default="resnet50")
    parser.add_argument("--image-size", type=int, default=200)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_config=TrainingConfig(architecture=args.architecture),
        image_size=args.image_size,
        device=args.device,
    )
