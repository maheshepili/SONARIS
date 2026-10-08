import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from src.models.train_detector import main


class TrainDetectorTests(unittest.TestCase):
    def _run_main(self, extra_args: list[str]) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "detector.yaml"
            config.write_text(
                "model: yolo11n.pt\nepochs: 1\nimgsz: 32\nbatch: 1\nseed: 42\n"
                f"prepared_data: {root.as_posix()}/data.yaml\nproject: {root.as_posix()}/runs\n",
                encoding="utf-8",
            )
            (root / "data.yaml").write_text("names:\n  0: Pipeline\n", encoding="utf-8")
            captured: dict = {}

            class FakeYOLO:
                def __init__(self, model: str):
                    captured["model"] = model

                def train(self, **kwargs: object) -> str:
                    captured["training_args"] = kwargs
                    return "trained"

            fake_ultralytics = ModuleType("ultralytics")
            fake_ultralytics.YOLO = FakeYOLO
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}), patch.object(
                sys, "argv", ["train_detector", "--config", str(config), *extra_args]
            ):
                main()
            return captured

    def test_uses_existing_default_training_name(self):
        captured = self._run_main([])
        self.assertEqual(captured["training_args"]["name"], "pipeline_yolo11n")

    def test_passes_custom_training_name_to_ultralytics(self):
        captured = self._run_main(["--name", "pipeline_finetune"])
        self.assertEqual(captured["training_args"]["name"], "pipeline_finetune")

