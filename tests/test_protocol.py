import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from run_baselines import ROOT, FEATURES, LABELS, SEEDS, SCORER, load_data, models


class ProtocolTests(unittest.TestCase):
    def test_data_and_feature_exclusions(self):
        data = load_data(ROOT / "short_form_brain_rot_cognitive_decay.csv")
        self.assertEqual(data.shape, (9000, 19))
        self.assertEqual(len(FEATURES), 6)
        self.assertNotIn("cognitive_decay_tier", FEATURES)
        self.assertNotIn("participant_id", FEATURES)
        self.assertEqual((data.cognitive_decay_tier == LABELS[-1]).sum(), 105)

    def test_budget_and_fold_local_scaling(self):
        for name, (estimator, grid) in models(42).items():
            if grid:
                self.assertEqual(len(list(ParameterGrid(grid))), 12, name)
            if name in ("logistic_regression", "rbf_svm"):
                sample = np.array([[0.] * 6, [2.] * 6, [1.] * 6, [3.] * 6])
                fitted = clone(estimator).fit(sample, [0, 0, 1, 1])
                scaler = fitted.named_steps["standardscaler"]
                np.testing.assert_allclose(scaler.mean_, [1.5] * 6)
                scaler.transform([[1000.] * 6])
                np.testing.assert_allclose(scaler.mean_, [1.5] * 6)

    def test_validation_scorer_accepts_dataset_string_labels(self):
        from sklearn.dummy import DummyClassifier
        X = np.zeros((8, 6))
        y = np.array(LABELS * 2)
        fitted = DummyClassifier(strategy="most_frequent").fit(X, y)
        expected = f1_score(y, fitted.predict(X), labels=LABELS, average="macro", zero_division=0)
        self.assertAlmostEqual(SCORER(fitted, X, y), expected)

    @unittest.skipUnless((ROOT / "results/baselines/summary.csv").exists(), "Full run not completed")
    def test_saved_evidence(self):
        out = ROOT / "results/baselines"
        data = load_data(ROOT / "short_form_brain_rot_cognitive_decay.csv")
        rows = pd.read_csv(out / "per_seed_metrics.csv")
        self.assertEqual(len(rows), 20)
        for seed in SEEDS:
            split = json.loads((out / str(seed) / "splits.json").read_text())
            train, test = set(split["train_row_positions"]), set(split["test_row_positions"])
            self.assertFalse(train & test)
            self.assertEqual(len(train), 7200)
            self.assertEqual(len(test), 1800)
            self.assertEqual(train | test, set(range(9000)))
            self.assertEqual((data.iloc[list(test)].cognitive_decay_tier == LABELS[-1]).sum(), 21)
            validation_rows = []
            for fold in split["cv_row_positions"]:
                a, b = set(fold["train"]), set(fold["validation"])
                self.assertFalse(a & b)
                self.assertEqual(a | b, train)
                self.assertFalse((a | b) & test)
                validation_rows.extend(b)
            self.assertEqual(len(validation_rows), len(set(validation_rows)))
            self.assertEqual(set(validation_rows), train)
            for name in models(seed):
                folder = out / str(seed) / name
                prediction = pd.read_csv(folder / "predictions.csv")
                metrics = json.loads((folder / "metrics.json").read_text())
                self.assertEqual(set(prediction.row_position), test)
                self.assertEqual(len(prediction), 1800)
                np.testing.assert_array_equal(prediction.true_label,
                    data.iloc[prediction.row_position].cognitive_decay_tier)
                np.testing.assert_array_equal(prediction.participant_id,
                    data.iloc[prediction.row_position].participant_id)
                self.assertAlmostEqual(metrics["macro_f1"], f1_score(
                    prediction.true_label, prediction.predicted_label, labels=LABELS,
                    average="macro", zero_division=0))
                self.assertEqual(np.array(metrics["confusion_matrix"]).sum(), 1800)
                if name != "majority":
                    self.assertEqual(len(pd.read_csv(folder / "cv_results.csv")), 12)


if __name__ == "__main__":
    unittest.main()
