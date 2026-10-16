"""Offline checks for alias selection, integrity failures and anonymous fetching."""
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.hub_artifacts import resolve_adapter, verify_adapter, download_part_a_adapter


class HubArtifactTests(unittest.TestCase):
    def test_pending_part_a_publication_does_not_download_main(self):
        manifest = {"repo_id": "Pritosh1/test", "revision": None,
                    "adapters": {"A1": {}}}
        with patch("src.common.hub_artifacts.load_part_a_manifest", return_value=manifest):
            with self.assertRaisesRegex(RuntimeError, "has not been published and pinned"):
                download_part_a_adapter("A1")

    def test_missing_a1_uses_its_own_checkpoint(self):
        manifest = {"adapters": {"A1": {}}}
        with patch("src.common.hub_artifacts.load_part_a_manifest", return_value=manifest), \
             patch("src.common.hub_artifacts.download_part_a_adapter", return_value=Path("snapshot/A1")) as fetch:
            self.assertEqual(resolve_adapter(Path("adapters/A1")), Path("snapshot/A1"))
            fetch.assert_called_once_with("A1")

    def test_missing_custom_path_does_not_substitute_a_model(self):
        with self.assertRaises(ValueError):
            resolve_adapter(Path("not-a-real-custom-adapter"))

    def test_no_adapter_keeps_base_model(self):
        self.assertIsNone(resolve_adapter(None))

    def test_download_alias_checks_weights_and_uses_no_token(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "best_sanitizer"
            folder.mkdir()
            files = {
                "adapter_config.json": json.dumps({"base_model_name_or_path": "test/base"}).encode(),
                "adapter_model.safetensors": b"test checkpoint bytes",
            }
            for name, data in files.items():
                (folder / name).write_bytes(data)
            entry = {"subfolder": "best_sanitizer", "base_model": "test/base",
                     "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
            manifest = {"repo_id": "Pritosh1/test", "revision": "fixed-commit",
                        "adapters": {"best_sanitizer": entry}}
            from unittest.mock import Mock
            downloader = Mock(return_value=str(root))
            hub = types.SimpleNamespace(snapshot_download=downloader)
            with patch("src.common.hub_artifacts.load_manifest", return_value=manifest), \
                 patch.dict("sys.modules", {"huggingface_hub": hub}):
                self.assertEqual(resolve_adapter(Path("adapters/A4")), folder)
                self.assertIs(downloader.call_args.kwargs["token"], False)
                self.assertEqual(downloader.call_args.kwargs["revision"], "fixed-commit")
            (folder / "adapter_model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_adapter(folder, entry)


if __name__ == "__main__":
    unittest.main()
