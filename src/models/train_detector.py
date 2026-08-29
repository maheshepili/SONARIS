"""Fine-tune a lightweight YOLO detector on prepared Pipeline labels."""

from __future__ import annotations

import argparse
from pathlib import Path


def load_detector_config(path: Path) -> dict[str, str]:
    """Read the deliberately flat project configuration without extra packages."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid detector configuration line: {line}")
        values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", help="Ultralytics model or local checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", default=None, help="e.g. 0 for CUDA, cpu for CPU")
    args = parser.parse_args()
    if not args.config.exists():
        raise SystemExit(f"Detector configuration not found: {args.config}")
    config = load_detector_config(args.config)
    data_path = args.data or Path(config["prepared_data"])
    if not data_path.exists():
        raise SystemExit(f"Prepared data not found: {data_path}. Run python -m src.data.prepare_dataset first.")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("ultralytics is not installed. Install requirements in a project-local environment.") from error
    training_args = {
        "data": str(data_path.resolve()),
        "epochs": args.epochs or int(config["epochs"]),
        "imgsz": args.imgsz or int(config["imgsz"]),
        "batch": args.batch or int(config["batch"]),
        "project": config["project"],
        "name": "pipeline_yolo11n",
        "exist_ok": True,
        "seed": int(config["seed"]),
    }
    if args.device is not None:
        training_args["device"] = args.device
    model = YOLO(args.model or config["model"])
    result = model.train(**training_args)
    print(result)


if __name__ == "__main__":
    main()
