"""Train the sonar reconstruction autoencoder on background patches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.anomaly.autoencoder import SonarAutoencoder


class SonarBackgroundPatches(Dataset):
    """Sample fixed-size patches that do not overlap labelled detector targets."""

    def __init__(self, images_dir: Path, labels_dir: Path, patch_size: int = 128, patches_per_image: int = 4, seed: int = 42) -> None:
        self.images = sorted(images_dir.glob("*.png"))
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.seed = seed
        if not self.images:
            raise ValueError(f"No PNG images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.images) * self.patches_per_image

    @staticmethod
    def _boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
        boxes = []
        if not label_path.exists():
            return boxes
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                _, xc, yc, bw, bh = map(float, fields)
            except ValueError:
                continue
            boxes.append(((xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height))
        return boxes

    @staticmethod
    def _overlap(a: tuple[int, int, int, int], b: tuple[float, float, float, float]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.images[index // self.patches_per_image]
        rng = random.Random(self.seed + index)
        with Image.open(image_path) as source:
            image = source.convert("L")
            width, height = image.size
            if width < self.patch_size or height < self.patch_size:
                image = image.resize((max(width, self.patch_size), max(height, self.patch_size)), Image.Resampling.BILINEAR)
                width, height = image.size
            boxes = self._boxes(self.labels_dir / f"{image_path.stem}.txt", width, height)
            max_x, max_y = width - self.patch_size, height - self.patch_size
            chosen = None
            for _ in range(40):
                x = rng.randint(0, max_x)
                y = rng.randint(0, max_y)
                candidate = (x, y, x + self.patch_size, y + self.patch_size)
                if not any(self._overlap(candidate, box) for box in boxes):
                    chosen = candidate
                    break
            if chosen is None:
                chosen = (rng.randint(0, max_x), rng.randint(0, max_y), 0, 0)
                chosen = (chosen[0], chosen[1], chosen[0] + self.patch_size, chosen[1] + self.patch_size)
            patch = image.crop(chosen)
            array = np.asarray(patch, dtype=np.float32) / 255.0
            return torch.from_numpy(array).unsqueeze(0)


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SonarBackgroundPatches(Path(args.images), Path(args.labels), args.patch_size, args.patches_per_image, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SonarAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)
        epoch_loss = running / len(dataset)
        history.append(epoch_loss)
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.6f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "patch_size": args.patch_size, "history": history}, output)
    report = {"device": str(device), "images": len(dataset.images), "training_patches": len(dataset), "epochs": args.epochs, "final_loss": history[-1], "checkpoint": str(output)}
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/processed/sonaris_detection/images/train"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/sonaris_detection/labels/train"))
    parser.add_argument("--output", type=Path, default=Path("reports/anomaly/sonar_autoencoder.pt"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
