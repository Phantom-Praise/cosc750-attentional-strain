import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from run_baselines import ROOT,LABELS,sha256
from run_tuning import candidates, WeightedBoosting


class TuningTests(unittest.TestCase):
    def test_boosting_string_label_weights_are_fold_local(self):
        from threadpoolctl import threadpool_limits
        X=np.random.default_rng(1).normal(size=(60,6))
        y=np.array([LABELS[0]]*30+[LABELS[1]]*15+[LABELS[2]]*10+[LABELS[3]]*5)
        with threadpool_limits(limits=1):
            for weight in ["balanced",{label:4 if label==LABELS[-1] else 1 for label in LABELS}]:
                fitted=WeightedBoosting(max_iter=2,class_weight=weight).fit(X,y)
                self.assertTrue(set(fitted.predict(X)).issubset(set(LABELS)))
                self.assertEqual(set(fitted.classes_),set(LABELS))
                if weight=="balanced":
                    self.assertAlmostEqual(fitted.fit_class_weights_[LABELS[-1]],60/(4*5))
                    self.assertAlmostEqual(fitted.fit_class_weights_[LABELS[0]],60/(4*30))
                else:
                    self.assertEqual(fitted.fit_class_weights_[LABELS[-1]],4)
                    self.assertEqual(fitted.fit_class_weights_[LABELS[0]],1)

    def test_frozen_candidate_budgets(self):
        first,second=candidates(),candidates()
        self.assertEqual(first,second)
        self.assertEqual(len(first["original_svm"]),12)
        for name in ["expanded_svm","gradient_boosting"]:
            self.assertEqual(len(first[name]),36)
            keys={json.dumps(p,sort_keys=True) for p in first[name]}
            self.assertEqual(len(keys),36)
        for config in first["original_svm"]:
            self.assertIn(config,first["expanded_svm"])

    @unittest.skipUnless((ROOT/"results/tuning/RESULTS.md").exists(),"Nested run not complete")
    def test_nested_separation_and_results(self):
        out=ROOT/"results/tuning"
        manifest=json.loads((out/"manifest.json").read_text())
        self.assertEqual(manifest["status"],"complete")
        self.assertEqual(manifest["script_sha256"],sha256(ROOT/"run_tuning.py"))
        self.assertEqual(manifest["model_script_sha256"],sha256(ROOT/"tuning_models.py"))
        self.assertEqual(manifest["candidate_sha256"],sha256(out/"candidates.json"))
        original=json.loads((ROOT/"results/baselines/42/splits.json").read_text())
        allowed,excluded=set(original["train_row_positions"]),set(original["test_row_positions"])
        covered=[]
        for fold in range(3):
            directory=out/f"fold_{fold}"
            split=json.loads((directory/"splits.json").read_text())
            a,b=set(split["train_row_positions"]),set(split["validation_row_positions"])
            self.assertFalse(a&b)
            self.assertEqual(a|b,allowed)
            self.assertFalse((a|b)&excluded)
            covered.extend(b)
            inner_valid=[]
            for inner in split["inner_folds"]:
                c,d=set(inner["train"]),set(inner["validation"])
                self.assertFalse(c&d)
                self.assertEqual(c|d,a)
                self.assertFalse((c|d)&b)
                inner_valid.extend(d)
            self.assertEqual(set(inner_valid),a)
            self.assertEqual(len(inner_valid),len(a))
            for name,count in [("original_svm",12),("expanded_svm",36),("gradient_boosting",36)]:
                self.assertEqual(len(pd.read_csv(directory/name/"cv_results.csv")),count)
        self.assertEqual(len(covered),len(allowed))
        self.assertEqual(set(covered),allowed)
        for name in candidates():
            pred=pd.read_csv(out/f"{name}_oof_predictions.csv")
            self.assertEqual(len(pred),7200)
            self.assertTrue(pred.row_position.is_unique)
            self.assertEqual(set(pred.row_position),allowed)
            measured=f1_score(pred.true_label,pred.predicted_label,labels=LABELS,average="macro",zero_division=0)
            saved=json.loads((out/f"{name}_pooled_metrics.json").read_text())
            self.assertAlmostEqual(measured,saved["macro_f1"])
        bundle=joblib.load(out/"model.joblib")
        self.assertEqual(set(bundle["training_row_positions"]),allowed)
        self.assertFalse(set(bundle["training_row_positions"])&excluded)
        self.assertFalse(json.loads((out/"final_model.json").read_text())["test_evaluated"])


if __name__=="__main__":
    unittest.main()
