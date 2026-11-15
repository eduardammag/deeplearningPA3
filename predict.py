"""CPU/offline inference for the delivered PA3 model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet18


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica imagens com o modelo entregue.")
    parser.add_argument("--imagens", required=True, type=Path, help="Pasta com imagens JPG.")
    parser.add_argument("--saida", required=True, type=Path, help="CSV de saída.")
    parser.add_argument("--pesos", type=Path, default=Path(__file__).with_name("modelo_entregue.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.pesos, map_location="cpu", weights_only=True)
    classes = checkpoint["class_names"]
    size = int(checkpoint["image_size"])
    mean = torch.tensor(checkpoint.get("mean", (0.485, 0.456, 0.406))).view(3, 1, 1)
    std = torch.tensor(checkpoint.get("std", (0.229, 0.224, 0.225))).view(3, 1, 1)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print(f"Parâmetros do modelo: {sum(p.numel() for p in model.parameters()):,}")

    files = sorted(p for p in args.imagens.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"Nenhuma imagem JPG, JPEG ou PNG encontrada em {args.imagens}")
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["image_id", "label"])
        with torch.inference_mode():
            for path in files:
                with Image.open(path) as source:
                    image = ImageOps.pad(source.convert("RGB"), (size, size), method=Image.Resampling.BILINEAR)
                tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0).permute(2, 0, 1)
                tensor = (tensor - mean) / std
                label = classes[int(model(tensor.unsqueeze(0)).argmax(1).item())]
                writer.writerow([path.stem, label])


if __name__ == "__main__":
    main()
