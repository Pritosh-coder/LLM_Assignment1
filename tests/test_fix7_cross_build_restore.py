from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import restore_c1_checkpoint


class Fix7CrossBuildRestoreTests(unittest.TestCase):
    def test_fix5_checkpoint_manifest_can_restore_into_fix7_root(self) -> None:
        source_build = "2026-09-14-fix5-resumable-staged-part-c"
        active_build = "2026-09-14-fix7-cross-build-c1-restore"
        payload = b"measured artifact\n"
        rel = "results/example.json"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            root.mkdir()
            (root / "KAGGLE_BUILD_ID").write_text(active_build + "\n", encoding="utf-8")
            archive = Path(td) / "c1.zip"
            manifest = {
                "build_id": source_build,
                "files": {
                    rel: {
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                },
            }
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("C1_CHECKPOINT_MANIFEST.json", json.dumps(manifest))
                zf.writestr(rel, payload)

            argv = [
                "restore_c1_checkpoint.py",
                "--archive",
                str(archive),
                "--root",
                str(root),
                "--source-build-id",
                source_build,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                restore_c1_checkpoint, "validate"
            ) as validate:
                restore_c1_checkpoint.main()

            self.assertEqual((root / rel).read_bytes(), payload)
            validate.assert_called_once_with(root.resolve(), expected_build_id=active_build)


if __name__ == "__main__":
    unittest.main()
