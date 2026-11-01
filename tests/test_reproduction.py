import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from strain.workflow import ROOT, DATA_FILE, MODEL_ORDER, baseline_definitions, obtain_data, proposed_candidates
from strain.config import FEATURES, LABELS, SEEDS
from tools.build_submission import build


class ReproductionTests(unittest.TestCase):
    def test_all_required_baselines_and_budgets(self):
        definitions = baseline_definitions(42)
        self.assertEqual(list(definitions), MODEL_ORDER)
        for name, (_, grid) in definitions.items():
            if name == "majority": self.assertIsNone(grid)
            else: self.assertEqual(len(list(ParameterGrid(grid))), 12, name)
        self.assertEqual(len(proposed_candidates()), 24)

    def test_data_not_downloaded_without_request(self):
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen") as request:
            with self.assertRaises(FileNotFoundError): obtain_data(Path(tmp)/DATA_FILE)
            request.assert_not_called()

    def test_corrupted_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/DATA_FILE
            path.write_text("invalid,data\n")
            with self.assertRaises(ValueError): obtain_data(path)

    def test_download_hash_verified_before_any_file_write(self):
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("../../"+DATA_FILE, "wrong file contents")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/DATA_FILE
            with patch("strain.workflow.urllib.request.urlopen", return_value=BytesIO(buffer.getvalue())):
                with self.assertRaises(ValueError): obtain_data(path, download=True)
            self.assertFalse(path.exists())

    def test_source_package_excludes_data_environment_and_execution_debris(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"source"; root.mkdir()
            (root/"reproduce.py").write_text("print('test')\n")
            (root/"README.md").write_text("test\n")
            (root/DATA_FILE).write_text("must not be included")
            (root/"strain").mkdir()
            (root/"strain/config.py").write_text("FEATURES = []\n")
            for folder in [".venv", "results", "tmp", "archive"]:
                (root/folder).mkdir(); (root/folder/"unwanted.py").write_text("unwanted")
            first = build(Path(tmp)/"first.zip", root)
            second = build(Path(tmp)/"second.zip", root)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with ZipFile(first) as archive:
                names = archive.namelist()
                self.assertIn("cosc750-stage3-code/reproduce.py", names)
                self.assertIn("cosc750-stage3-code/strain/config.py", names)
                self.assertIn("cosc750-stage3-code/PACKAGE_MANIFEST.json", names)
                self.assertFalse(any(DATA_FILE in n or '/.venv/' in n or '/results/' in n or '/tmp/' in n or '/archive/' in n for n in names))

    @unittest.skipUnless((ROOT/"results/submission/baseline_table.csv").exists(), "Submission runner not completed")
    def test_portable_baseline_evidence(self):
        out = ROOT/"results/submission"
        m = json.loads((out/"manifest.json").read_text())
        self.assertEqual(m["status"], "complete")
        self.assertEqual(m["features"], FEATURES)
        rows = pd.read_csv(out/"baseline_per_run.csv")
        self.assertEqual(len(rows), 25)
        self.assertEqual(len(pd.read_csv(out/"baseline_per_class.csv")), 100)
        validation_rows = {}
        for seed in SEEDS:
            split = json.loads((out/f"baseline/{seed}/splits.json").read_text())
            train, test = set(split["train_row_positions"]), set(split["test_row_positions"])
            self.assertFalse(train & test)
            self.assertEqual(train | test, set(range(9000)))
            self.assertEqual(len(test), 1800)
            for name in MODEL_ORDER:
                folder = out/f"baseline/{seed}"/name
                pred = pd.read_csv(folder/"predictions.csv")
                self.assertEqual(set(pred.row_position), test)
                value = f1_score(pred.true_label, pred.predicted_label, labels=LABELS,
                                 average="macro", zero_division=0)
                saved = json.loads((folder/"metrics.json").read_text())
                self.assertAlmostEqual(value, saved["macro_f1"])
                if name != "majority": self.assertEqual(len(pd.read_csv(folder/"cv_results.csv")), 12)

    @unittest.skipUnless((ROOT/"results/submission/proposed_validation_table.csv").exists(), "Optional proposed reproduction not completed")
    def test_proposed_evidence_and_exclusion(self):
        out = ROOT/"results/submission"
        original = json.loads((out/"baseline/42/splits.json").read_text())
        training = set(original["train_row_positions"])
        excluded = set(original["test_row_positions"])
        covered = []
        for fold in range(3):
            folder = out/f"proposed_validation/fold_{fold}"
            split = json.loads((folder/"splits.json").read_text())
            a, b = set(split["train_row_positions"]), set(split["validation_row_positions"])
            self.assertFalse(a & b)
            self.assertEqual(a | b, training)
            self.assertFalse((a | b) & excluded)
            covered.extend(b)
            for inner in split["inner_folds"]:
                c, d = set(inner["train"]), set(inner["validation"])
                self.assertFalse(c & d)
                self.assertEqual(c | d, a)
            for name, budget in [("rbf_svm", 12), ("calibrated_svm", 24)]:
                predictions = pd.read_csv(folder/name/"predictions.csv")
                self.assertEqual(set(predictions.row_position), b)
                self.assertEqual(len(pd.read_csv(folder/name/"cv_results.csv")), budget)
        self.assertEqual(len(covered), 7200)
        self.assertEqual(set(covered), training)


if __name__ == "__main__": unittest.main()
