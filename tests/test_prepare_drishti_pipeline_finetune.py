import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.prepare_drishti_pipeline_finetune import (
    prepare_drishti_pipeline_finetune,
    remap_pipeline_labels,
    source_timestamp,
    split_timestamp_groups,
)


class PrepareDrishtiPipelineFinetuneTests(unittest.TestCase):
    def test_extracts_timestamp_from_pipeline_and_background_crop_names(self):
        self.assertEqual(source_timestamp(Path("pipe_1693569376.780_x1500.jpg")), "1693569376.780")
        self.assertEqual(source_timestamp(Path("drishti_bg_1693570064.959_x500.jpg")), "1693570064.959")
        self.assertIsNone(source_timestamp(Path("mine_0001_2021.jpg")))

    def test_remaps_every_pipeline_label_to_zero(self):
        self.assertEqual(
            remap_pipeline_labels("1 0.5 0.5 0.2 0.2\n1 0.2 0.3 0.1 0.1\n"),
            "0 0.5 0.5 0.2 0.2\n0 0.2 0.3 0.1 0.1\n",
        )
        with self.assertRaisesRegex(ValueError, "only source class 1"):
            remap_pipeline_labels("1 0.5 0.5 0.2 0.2\n2 0.2 0.3 0.1 0.1\n")

    def test_timestamp_groups_are_deterministic_and_never_split(self):
        timestamps = {"100.000", "200.000", "300.000", "400.000", "500.000"}
        first = split_timestamp_groups(timestamps)
        self.assertEqual(first, split_timestamp_groups(timestamps))
        self.assertEqual(sum(value == "train" for value in first.values()), 4)
        self.assertEqual(sum(value == "val" for value in first.values()), 1)

    def test_excludes_external_validation_timestamp_and_keeps_crop_group_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            images, labels = train / "images", train / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            external_validation = root / "external_validation"
            external_validation.mkdir()

            def add_image(timestamp: str, offset: int) -> str:
                name = f"bg_{timestamp}_x{offset}.jpg"
                Image.new("L", (10, 10)).save(images / name)
                (labels / f"bg_{timestamp}_x{offset}.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                return name

            leaked = add_image("100.000", 0)
            group_one = [add_image("200.000", 0), add_image("200.000", 500)]
            group_two = add_image("300.000", 0)
            Image.new("L", (10, 10)).save(external_validation / "drishti_bg_100.000_x500.jpg")

            output = root / "output"
            manifest = prepare_drishti_pipeline_finetune(train, external_validation, output)

            self.assertIn(leaked, [record["image"] for record in manifest["excluded_images"]])
            record_splits = {record["image"]: record["split"] for record in manifest["images"]}
            self.assertEqual(record_splits[group_one[0]], record_splits[group_one[1]])
            self.assertIn(group_two, record_splits)
            for split in ("train", "val"):
                for label in (output / "labels" / split).glob("*.txt"):
                    self.assertTrue(label.read_text(encoding="utf-8").startswith("0 "))
            self.assertEqual(
                (output / "data.yaml").read_text(encoding="utf-8"),
                f"path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: Pipeline\n",
            )

    def test_include_negatives_selects_only_pure_non_pipeline_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            images, labels = train / "images", train / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            external_validation = root / "external_validation"
            external_validation.mkdir()

            def add_image(timestamp: str, label: str) -> str:
                name = f"bg_{timestamp}_x0.jpg"
                Image.new("L", (10, 10)).save(images / name)
                (labels / f"bg_{timestamp}_x0.txt").write_text(label, encoding="utf-8")
                return name

            positive = add_image("100.000", "1 0.5 0.5 0.2 0.2\n")
            negative = add_image("200.000", "2 0.5 0.5 0.2 0.2\n3 0.4 0.4 0.1 0.1\n")
            mixed = add_image("300.000", "1 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n")
            empty = add_image("400.000", "\n")
            leaked_negative = add_image("500.000", "4 0.5 0.5 0.2 0.2\n")
            Image.new("L", (10, 10)).save(external_validation / "bg_500.000_x500.jpg")

            output = root / "output"
            manifest = prepare_drishti_pipeline_finetune(
                train, external_validation, output, include_negatives=True
            )

            records = {record["image"]: record for record in manifest["images"]}
            self.assertEqual(records[positive]["type"], "positive")
            self.assertEqual(records[negative]["type"], "negative")
            self.assertEqual(records[negative]["source_class_ids"], [2, 3])
            negative_split = records[negative]["split"]
            self.assertEqual(
                (output / "labels" / negative_split / "bg_200.000_x0.txt").read_text(encoding="utf-8"),
                "",
            )
            self.assertNotIn(mixed, records)
            self.assertNotIn(empty, records)
            self.assertNotIn(leaked_negative, records)
            self.assertEqual(
                manifest["image_type_split_counts"]["positive"][records[positive]["split"]], 1
            )
            self.assertEqual(manifest["image_type_split_counts"]["negative"][negative_split], 1)
            self.assertEqual(
                manifest["negative_source_class_counts"][negative_split], {"2": 1, "3": 1, "4": 0}
            )
            exclusions = {entry["image"]: entry["reason"] for entry in manifest["excluded_images"]}
            self.assertEqual(exclusions[mixed], "not_pipeline_positive_or_pure_non_pipeline_negative")
            self.assertEqual(exclusions[empty], "not_pipeline_positive_or_pure_non_pipeline_negative")
            self.assertEqual(exclusions[leaked_negative], "external_validation_timestamp_overlap")
