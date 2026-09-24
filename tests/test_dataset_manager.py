from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "dataset_manager.py"


class DatasetManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.bin"
        self.source.write_bytes(b"verified dataset payload")

        fake_gdown = types.ModuleType("gdown")

        def fake_download(**kwargs: object) -> str:
            output = Path(str(kwargs["output"]))
            output.write_bytes(self.source.read_bytes())
            return str(output)

        fake_gdown.download = fake_download  # type: ignore[attr-defined]
        sys.modules["gdown"] = fake_gdown
        self.addCleanup(sys.modules.pop, "gdown", None)

        spec = importlib.util.spec_from_file_location(
            "dataset_manager_under_test", MODULE_PATH
        )
        assert spec and spec.loader
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(self.module)

        self.module.DATA_ROOT = self.root / "datasets"
        self.module.MANIFEST = self.root / ".dataset-manifest.tsv"
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.module.MANIFEST.write_text(
            f"{digest}\t{self.source.stat().st_size}\tfile-id\t"
            "article_thin_sections/sample.bin\n",
            encoding="utf-8",
        )

    def entries(self):
        return self.module.load_manifest("app")

    def test_download_promotes_only_verified_file(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            result = self.module.download(self.entries(), False)
        target = (
            self.module.DATA_ROOT
            / "article_thin_sections"
            / "sample.bin"
        )
        self.assertEqual(result, 0)
        self.assertEqual(target.read_bytes(), self.source.read_bytes())
        self.assertIn("Verified SHA-256", output.getvalue())

    def test_invalid_file_requires_explicit_replacement(self) -> None:
        target = (
            self.module.DATA_ROOT
            / "article_thin_sections"
            / "sample.bin"
        )
        target.parent.mkdir(parents=True)
        target.write_bytes(b"invalid")
        self.assertEqual(self.module.download(self.entries(), False), 3)
        self.assertEqual(target.read_bytes(), b"invalid")

        self.assertEqual(self.module.download(self.entries(), True), 0)
        backups = list(target.parent.glob("sample.bin.invalid-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"invalid")

    def test_verify_reports_missing_and_valid(self) -> None:
        self.assertEqual(self.module.verify(self.entries()), 2)
        target = (
            self.module.DATA_ROOT
            / "article_thin_sections"
            / "sample.bin"
        )
        target.parent.mkdir(parents=True)
        target.write_bytes(self.source.read_bytes())
        self.assertEqual(self.module.verify(self.entries()), 0)

    def test_manifest_rejects_parent_traversal(self) -> None:
        self.module.MANIFEST.write_text(
            f"{'0' * 64}\t1\tfile-id\t../outside.bin\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            self.module.load_manifest("all")


if __name__ == "__main__":
    unittest.main()
