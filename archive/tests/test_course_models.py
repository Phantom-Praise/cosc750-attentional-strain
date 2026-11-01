import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from course_models import HistogramNaiveBayes, minimum_risk_decision

ROOT = Path(__file__).resolve().parents[1]


class CourseModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[45,55],[55,60],[65,80],[75,70],
                           [55,85],[65,90],[85,95],[95,65]])
        self.y = np.array(["Fail"]*4 + ["Pass"]*4)
        self.edges = [[40,60,80,100], [50,75,100]]

    def test_lecture_unsmoothed_example(self):
        model = HistogramNaiveBayes(alpha=0, bin_edges=self.edges).fit(self.X, self.y)
        np.testing.assert_allclose(model.predict_proba([[70,78]]), [[.4,.6]])
        np.testing.assert_allclose(model.predict_proba([[90,62]]), [[0,1]])
        self.assertEqual(model.bin_density_[0][0,0], .025)

    def test_lecture_smoothed_example(self):
        model = HistogramNaiveBayes(alpha=1, bin_edges=self.edges).fit(self.X, self.y)
        np.testing.assert_allclose(model.predict_proba([[90,62]]), [[.4,.6]])
        np.testing.assert_allclose(model.predict_proba([[70,78]]), [[3/7,4/7]])
        for edges, density, counts in zip(model.bin_edges_, model.bin_density_, model.bin_count_):
            np.testing.assert_allclose((density*np.diff(edges)).sum(axis=1), 1)
            np.testing.assert_allclose(counts.sum(axis=1), [4,4])

    def test_boundary_and_finite_probabilities(self):
        model = HistogramNaiveBayes(n_bins=10, alpha=1).fit(self.X, self.y)
        saved_edges = [e.copy() for e in model.bin_edges_]
        prob = model.predict_proba([[-1000,1000], [1000,-1000]])
        self.assertTrue(np.isfinite(prob).all())
        np.testing.assert_allclose(prob.sum(axis=1), 1)
        for before, after in zip(saved_edges, model.bin_edges_):
            np.testing.assert_array_equal(before, after)

    def test_equal_and_unequal_cost_decisions(self):
        posterior = [[.8,.2]]
        self.assertEqual(minimum_risk_decision(posterior)[0], 0)
        # Action 0 misses class 1 at cost 5; action 1 falsely flags class 0 at cost 1.
        self.assertEqual(minimum_risk_decision(posterior, [[0,5],[1,0]])[0], 1)

    @unittest.skipUnless((ROOT/"results/course_extension/summary.csv").exists(), "Extension not run")
    def test_extension_evidence(self):
        from run_baselines import LABELS, SEEDS, FEATURES, sha256
        out = ROOT/"results/course_extension"
        manifest = json.loads((out/"manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["model_script_sha256"],sha256(ROOT/"course_models.py"))
        self.assertEqual(manifest["extension_script_sha256"],sha256(ROOT/"run_course_extension.py"))
        metrics = pd.read_csv(out/"per_seed_metrics.csv")
        self.assertEqual(len(metrics),25)
        class_metrics = pd.read_csv(out/"per_class_metrics.csv")
        self.assertEqual(len(class_metrics),100)
        for seed in SEEDS:
            seed_dir = out/str(seed)
            model = json.loads((seed_dir/"histogram_nb/fitted_histograms.json").read_text())
            np.testing.assert_allclose(sum(model["class_priors"]),1)
            self.assertEqual(sum(model["class_counts"]),7200)
            cv = pd.read_csv(seed_dir/"histogram_nb/cv_results.csv")
            self.assertEqual(len(cv),12)
            split_file = ROOT/f"results/baselines/{seed}/splits.json"
            split = json.loads(split_file.read_text())
            self.assertEqual(json.loads((seed_dir/"split_reference.json").read_text())
                             ["baseline_split_sha256"],sha256(split_file))
            for feature in FEATURES:
                histogram=model["features"][feature]
                densities=np.array(histogram["densities"])
                np.testing.assert_allclose((densities*np.diff(histogram["edges"])).sum(axis=1),1)
                np.testing.assert_array_equal(np.array(histogram["counts"]).sum(axis=1),model["class_counts"])
            for name in metrics.model.unique():
                predictions=pd.read_csv(seed_dir/name/"predictions_and_scores.csv")
                np.testing.assert_array_equal(predictions.row_position,split["test_row_positions"])
                if name!="histogram_nb":
                    old=pd.read_csv(ROOT/f"results/baselines/{seed}/{name}/predictions.csv")
                    np.testing.assert_array_equal(old.predicted_label,predictions.predicted_label)
                if name!="rbf_svm":
                    probabilities=predictions[["score_normal","score_mild","score_severe","score_acute"]]
                    np.testing.assert_allclose(probabilities.sum(axis=1),1)
                risk=np.mean(predictions.true_label!=predictions.predicted_label)
                row=metrics.loc[(metrics.seed==seed)&(metrics.model==name)].iloc[0]
                self.assertAlmostEqual(risk,row.empirical_zero_one_risk)


if __name__ == "__main__":
    unittest.main()
