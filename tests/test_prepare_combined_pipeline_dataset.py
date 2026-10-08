import tempfile
import unittest
from pathlib import Path

from src.data.prepare_combined_pipeline_dataset import prepare_combined_pipeline_dataset


class PrepareCombinedPipelineDatasetTests(unittest.TestCase):
    def test_combines_only_train_sources_and_preserves_labels_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sonaris = root / "sonaris"
            drishti = root / "drishti"

            def add_pair(dataset: Path, split: str, name: str, label: str) -> None:
                images = dataset / "images" / split
                labels = dataset / "labels" / split
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                (images / name).write_bytes(f"image-{dataset.name}-{split}-{name}".encode())
                (labels / f"{Path(name).stem}.txt").write_text(label, encoding="utf-8")

            add_pair(sonaris, "train", "shared.png", "0 0.5 0.5 0.2 0.2\n")
            add_pair(sonaris, "train", "sonaris_only.png", "0 0.4 0.4 0.1 0.1\n")
            add_pair(sonaris, "val", "sonaris_val.png", "0 0.3 0.3 0.1 0.1\n")
            add_pair(drishti, "train", "shared.png", "0 0.6 0.6 0.2 0.2\n")
            add_pair(drishti, "train", "drishti_only.png", "")

            output = root / "combined"
            manifest = prepare_combined_pipeline_dataset(sonaris, drishti, output)

            records = manifest["images"]
            self.assertEqual(len(records), 4)
            self.assertEqual({record["source_dataset"] for record in records}, {"sonaris", "drishti"})
            self.assertTrue(all(record["source_split"] == "train" for record in records))
            self.assertNotIn("sonaris_val.png", {record["output_image"] for record in records})
            self.assertEqual(len({record["output_image"] for record in records}), 4)
            self.assertEqual(len({Path(record["output_image"]).stem for record in records}), 4)

            shared_records = [record for record in records if record["source_image"] == "shared.png"]
            self.assertEqual(
                {record["output_image"] for record in shared_records},
                {"shared.png", "drishti_1_shared.png"},
            )
            for record in records:
                source_root = sonaris if record["source_dataset"] == "sonaris" else drishti
                self.assertEqual(
                    (output / "labels" / "train" / record["output_label"]).read_text(encoding="utf-8"),
                    (source_root / "labels" / "train" / record["source_label"]).read_text(encoding="utf-8"),
                )

            self.assertEqual(manifest["classes"], {"0": "Pipeline"})
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(manifest["source_counts"], {"sonaris": 2, "drishti": 2})
            self.assertEqual(manifest["output_counts"], {"train_images": 4, "train_labels": 4})
            self.assertEqual(manifest["validation_reference"]["source_dataset"], "sonaris")
            self.assertEqual(manifest["validation_reference"]["image_count"], 1)
            self.assertEqual(
                (output / "data.yaml").read_text(encoding="utf-8"),
                f"path: {output.resolve().as_posix()}\n"
                "train: images/train\n"
                f"val: {(sonaris / 'images' / 'val').resolve().as_posix()}\n"
                "names:\n  0: Pipeline\n",
            )


if __name__ == "__main__":
    unittest.main()
