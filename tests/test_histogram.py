import unittest
import numpy as np
from strain.histogram import HistogramNaiveBayes, minimum_risk_decision


class HistogramTests(unittest.TestCase):

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
