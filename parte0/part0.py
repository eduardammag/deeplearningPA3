"""Supervised baselines for Part 0 of the PA3 assignment."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_classes(classes_csv: Path) -> list[str]:
    with classes_csv.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"idx", "label"}.issubset(rows[0]):
        raise ValueError(f"{classes_csv} deve conter as colunas idx,label.")
    rows.sort(key=lambda row: int(row["idx"]))
    indices = [int(row["idx"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"Os índices em {classes_csv} devem começar em zero e ser contíguos.")
    return [row["label"] for row in rows]


def read_labels(labels_csv: Path, class_names: list[str]) -> list[tuple[str, int]]:
    name_to_index = {name: index for index, name in enumerate(class_names)}
    with labels_csv.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"image_id", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{labels_csv} deve conter as colunas image_id,label.")
        rows: list[tuple[str, int]] = []
        seen_ids: set[str] = set()
        for line, row in enumerate(reader, start=2):
            image_id = row["image_id"].strip()
            raw_label = row["label"].strip()
            if not image_id or not raw_label:
                raise ValueError(f"ID ou rótulo vazio em {labels_csv}, linha {line}.")
            if image_id in seen_ids:
                raise ValueError(f"ID duplicado {image_id!r} em {labels_csv}.")
            seen_ids.add(image_id)
            if image_id.lower().endswith(".jpg"):
                image_id = image_id[:-4]
            try:
                label_index = int(raw_label)
            except ValueError:
                if raw_label not in name_to_index:
                    raise ValueError(f"Classe desconhecida {raw_label!r} em {labels_csv}, linha {line}.")
                label_index = name_to_index[raw_label]
            if not 0 <= label_index < len(class_names):
                raise ValueError(f"Índice de classe inválido {label_index} em {labels_csv}, linha {line}.")
            rows.append((image_id, label_index))
    if not rows:
        raise ValueError(f"Nenhum exemplo encontrado em {labels_csv}.")
    return rows


class CameraTrapDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, images_dir: Path, labels_csv: Path, class_names: list[str], image_size: int):
        self.images_dir = images_dir
        self.rows = read_labels(labels_csv, class_names)
        self.image_size = image_size
        self.mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
        missing = [
            image_id
            for image_id, _ in self.rows
            if not (images_dir / f"{image_id}.jpg").is_file()
        ]
        if missing:
            sample = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} imagens referenciadas por {labels_csv} não estão em "
                f"{images_dir}; exemplos: {sample}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_id, label = self.rows[index]
        with Image.open(self.images_dir / f"{image_id}.jpg") as source:
            image = ImageOps.pad(
                source.convert("RGB"),
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std, label


class SmallCNN(nn.Module):
    """A compact CNN trained from random initialization."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def make_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "small_cnn":
        return SmallCNN(num_classes)
    if model_name == "resnet18":
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise RuntimeError(
                "A versão ResNet-18 requer torchvision. Instale as dependências com "
                "`python -m pip install -r requirements.txt`."
            ) from error
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Modelo desconhecido: {model_name}")


def evaluate(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> tuple[float, float, float, list[int], list[int]]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return (
        total_loss / len(loader.dataset),
        f1_score(actual, predicted, average="macro", zero_division=0),
        accuracy_score(actual, predicted),
        actual,
        predicted,
    )


def write_history_plot(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Treino")
    axes[0].plot(epochs, history["val_loss"], label="Validação")
    axes[0].set(title="Perda", xlabel="Época", ylabel="Cross-entropy")
    axes[1].plot(epochs, history["train_macro_f1"], label="Treino")
    axes[1].plot(epochs, history["val_macro_f1"], label="Validação")
    axes[1].set(title="F1 macro", xlabel="Época", ylabel="F1 macro")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_model(
    model_name: str,
    datasets: dict[str, CameraTrapDataset],
    class_names: list[str],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    image_size: int,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(model_name, len(class_names)).to(device)
    if model_name == "resnet18":
        learning_rate = 1e-4
    else:
        learning_rate = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }
    best_f1 = -1.0
    best_val_loss = float("inf")
    best_checkpoint = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        train_loss, train_f1, _, _, _ = evaluate(model, loaders["train"], device)
        val_loss, val_f1, _, _, _ = evaluate(model, loaders["val"], device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(train_f1)
        history["val_macro_f1"].append(val_f1)
        print(
            f"{model_name} época {epoch:03d}/{epochs} | "
            f"perda treino/val={train_loss:.4f}/{val_loss:.4f} | "
            f"F1 macro treino/val={train_f1:.4f}/{val_f1:.4f}"
        )
        if val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_val_loss):
            best_f1 = val_f1
            best_val_loss = val_loss
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": image_size,
                    "mean": MEAN,
                    "std": STD,
                    "seed": seed,
                    "best_epoch": epoch,
                },
                best_checkpoint,
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    if model_name == "resnet18":
        write_history_plot(history, output_dir / "curva_treino_validacao.png")

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_f1, test_accuracy, actual, predicted = evaluate(
        model, loaders["test_public"], device
    )
    metrics = {
        "model": model_name,
        "best_epoch": checkpoint["best_epoch"],
        "test_public_macro_f1": test_f1,
        "test_public_accuracy": test_accuracy,
        "test_public_loss": test_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "num_test_public": len(actual),
        "class_names": class_names,
    }
    with (output_dir / "test_public_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(
        f"{model_name} | melhor época pelo val: {checkpoint['best_epoch']} | "
        f"F1 macro test_public={test_f1:.4f} | acurácia test_public={test_accuracy:.4f}"
    )
    if model_name == "resnet18":
        matrix = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))
        with (output_dir / "matriz_confusao.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["classe", *class_names])
            writer.writerows(
                [class_names[index], *row.tolist()] for index, row in enumerate(matrix)
            )
        figure, axis = plt.subplots(figsize=(12, 10))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=range(len(class_names)),
            yticks=range(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Classe prevista",
            ylabel="Classe real",
            title="ResNet-18 — matriz de confusão no test_public",
        )
        plt.setp(axis.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / "matriz_confusao.png", dpi=160)
        plt.close(figure)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina as duas baselines supervisionadas da Parte 0."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("pa3_dados"))
    parser.add_argument("--output-dir", type=Path, default=Path("parte0/resultados"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=("both", "small_cnn", "resnet18"),
        default="both",
        help="Permite executar uma versão individual ou as duas em sequência.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("--epochs e --batch-size devem ser positivos; --image-size deve ser >= 32.")
    return args


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    classes = load_classes(data_dir / "classes.csv")
    split_files = {
        "train": (data_dir / "images" / "train", data_dir / "labels" / "rotulos_5.csv"),
        "val": (data_dir / "images" / "val", data_dir / "labels" / "val.csv"),
        "test_public": (
            data_dir / "images" / "test_public",
            data_dir / "labels" / "test_public.csv",
        ),
    }
    missing = [str(labels) for _, labels in split_files.values() if not labels.is_file()]
    missing.extend(str(images) for images, _ in split_files.values() if not images.is_dir())
    if missing:
        raise FileNotFoundError(
            "Dados necessários ausentes: "
            + ", ".join(missing)
            + ". Extraia o pacote completo pa3_dados, incluindo labels/ e images/val."
        )
    datasets = {
        split: CameraTrapDataset(images, labels, classes, args.image_size)
        for split, (images, labels) in split_files.items()
    }
    if len(datasets["train"]) != 105:
        raise ValueError(
            f"A Parte 0 requer os 105 exemplos de rotulos_5.csv; "
            f"foram encontrados {len(datasets['train'])}."
        )
    if len(datasets["val"]) != 202:
        print(f"Aviso: o val tem {len(datasets['val'])} exemplos (o pacote informa 202).")
    model_names = ("small_cnn", "resnet18") if args.model == "both" else (args.model,)
    results = []
    for model_name in model_names:
        results.append(
            train_model(
                model_name,
                datasets,
                classes,
                args.output_dir / model_name,
                args.epochs,
                args.batch_size,
                args.image_size,
                args.seed,
            )
        )
    with (args.output_dir / "resumo_test_public.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
