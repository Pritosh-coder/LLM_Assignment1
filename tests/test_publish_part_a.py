"""Fail before network access when a shard is incomplete or contains changed bytes."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.publish_part_a import stage_shards, locate_run_file


class PartAArchiveTests(unittest.TestCase):
    def test_original_and_split_zip_paths(self):
        for path, expected in [
            ("adapters/A1/adapter_config.json", ("A1", "adapter_config.json")),
            ("Parta_3_1/A5/adapter_model.safetensors", ("A5", "adapter_model.safetensors")),
            ("partA_3_2/A6/tokenizer.json", ("A6", "tokenizer.json")),
            ("A6/merges.txt", ("A6", "merges.txt")),
        ]:
            self.assertEqual(locate_run_file(path), expected)

    def test_rejects_unsafe_paths(self):
        for path in ["../A5/adapter_config.json", "/A6/adapter_config.json", "A5/A6/adapter_config.json"]:
            with self.assertRaises(ValueError):
                locate_run_file(path)

    def fixture(self, root, corrupt=False, omit=False):
        config = json.dumps({"base_model_name_or_path": "test/base"}).encode()
        weights = b"original-weights"
        files = {"adapter_config.json": config, "adapter_model.safetensors": weights}
        entry = {"subfolder": "A1", "base_model": "test/base",
                 "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
        archive = root / "part.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name, data in files.items():
                if omit and name == "adapter_model.safetensors":
                    continue
                z.writestr("adapters/A1/" + name,
                           b"changed" if corrupt and name.endswith("safetensors") else data)
        return archive, {"adapters": {"A1": entry}}

    def test_stages_only_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root)
            self.assertEqual(stage_shards([archive], root / "stage", manifest), ["A1"])

    def test_rejects_missing_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root, omit=True)
            with self.assertRaisesRegex(ValueError, "Missing checkpoint files"):
                stage_shards([archive], root / "stage", manifest)

    def test_rejects_changed_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root, corrupt=True)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                stage_shards([archive], root / "stage", manifest)


if __name__ == "__main__":
    unittest.main()
