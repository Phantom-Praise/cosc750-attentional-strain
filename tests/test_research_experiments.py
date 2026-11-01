import json
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from run_baselines import ROOT,LABELS,sha256
from research_models import OrdinalLogistic,OversampledSVM,CalibratedAcuteSVM
from run_research_experiments import configurations


class ResearchExperimentTests(unittest.TestCase):
    def test_candidate_budgets(self):
        first,second=configurations(),configurations()
        self.assertEqual(first,second)
        self.assertEqual(len(first),6)
        for name,settings in first.items():
            self.assertEqual(len(settings),24,name)
            self.assertEqual(len({json.dumps(p,sort_keys=True) for p in settings}),24,name)

    def test_ordinal_order_probability_and_convergence(self):
        X=np.linspace(-3,3,240).reshape(-1,1)
        truth=np.digitize(X[:,0],[-1.5,0,1.5])
        y=np.array(LABELS)[truth]
        with threadpool_limits(limits=1):model=OrdinalLogistic(alpha=1).fit(X,y)
        self.assertTrue(model.optimization_["success"])
        self.assertTrue((np.diff(model.thresholds_)>0).all())
        p=model.predict_proba(X)
        np.testing.assert_allclose(p.sum(axis=1),1)
        self.assertTrue((p>=0).all())
        prediction=np.array([LABELS.index(v) for v in model.predict(X)])
        self.assertTrue((np.diff(prediction)>=0).all())
        self.assertGreater(np.mean(prediction==truth),.95)

    def test_sampling_is_fit_only_and_scaling_precedes_sampling(self):
        X=np.random.default_rng(1).normal(size=(80,6))
        y=np.array([LABELS[0]]*40+[LABELS[1]]*20+[LABELS[2]]*12+[LABELS[3]]*8)
        original=X.copy()
        with threadpool_limits(limits=1):
            for method in ["random","smote"]:
                fitted=OversampledSVM(sampling=method,multiplier=4).fit(X,y)
                np.testing.assert_allclose(fitted.model_.named_steps['standardscaler'].mean_,X.mean(axis=0))
                sampler=fitted.model_.steps[1][1]
                self.assertEqual(sampler.sampling_strategy_[LABELS[-1]],24)
                self.assertEqual(len(fitted.predict(X)),80)
                np.testing.assert_array_equal(X,original)

    def test_calibration_scalers_fit_only_calibration_training(self):
        X=np.random.default_rng(2).normal(size=(120,6))
        y=np.array(LABELS*30)
        with threadpool_limits(limits=1):model=CalibratedAcuteSVM().fit(X,y)
        folds=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(X,y))
        for member,(train,validation) in zip(model.model_.calibrated_classifiers_,folds):
            scaler=member.estimator.named_steps['standardscaler']
            np.testing.assert_allclose(scaler.mean_,X[train].mean(axis=0))
            self.assertFalse(set(train)&set(validation))
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1),1)

    def test_acute_rule_has_explicit_multiclass_fallback(self):
        model=CalibratedAcuteSVM(cutoff=.35)
        model.classes_=np.array(LABELS)
        model.predict_proba=lambda X:np.array([[.5,.05,.05,.4],[.1,.6,.1,.2]])
        np.testing.assert_array_equal(model.predict([[0],[0]]),[LABELS[-1],LABELS[1]])
        model.cutoff=None
        np.testing.assert_array_equal(model.predict([[0],[0]]),[LABELS[0],LABELS[1]])

    @unittest.skipUnless((ROOT/'results/research_experiments/RESULTS.md').exists(),'Study not complete')
    def test_complete_evidence(self):
        out=ROOT/'results/research_experiments'
        m=json.loads((out/'manifest.json').read_text())
        self.assertEqual(m['status'],'complete')
        self.assertFalse(m['test_evaluated'])
        self.assertEqual(m['script_sha256'],sha256(ROOT/'run_research_experiments.py'))
        self.assertEqual(m['model_script_sha256'],sha256(ROOT/'research_models.py'))
        self.assertEqual(m['candidate_sha256'],sha256(out/'candidates.json'))
        original=json.loads((ROOT/'results/baselines/42/splits.json').read_text())
        allowed=set(original['train_row_positions']);excluded=set(original['test_row_positions'])
        old=pd.read_csv(ROOT/'results/tuning/original_svm_oof_predictions.csv')
        current=pd.read_csv(out/'original_svm_oof_predictions.csv')
        np.testing.assert_array_equal(old.predicted_label,current.predicted_label)
        for name in ['original_svm',*configurations()]:
            pred=pd.read_csv(out/f'{name}_oof_predictions.csv')
            self.assertEqual(len(pred),7200)
            self.assertTrue(pred.row_position.is_unique)
            self.assertEqual(set(pred.row_position),allowed)
            self.assertFalse(set(pred.row_position)&excluded)
            score=f1_score(pred.true_label,pred.predicted_label,labels=LABELS,average='macro',zero_division=0)
            self.assertAlmostEqual(score,json.loads((out/f'{name}_pooled_metrics.json').read_text())['macro_f1'])
        for fold in range(3):
            split=out/f'fold_{fold}/splits.json'
            self.assertEqual(json.loads(split.read_text()),json.loads(
                (ROOT/f'results/tuning/fold_{fold}/splits.json').read_text()))
            for name in configurations():
                folder=out/f'fold_{fold}'/name
                self.assertEqual(len(pd.read_csv(folder/'cv_results.csv')),24)
                pred=pd.read_csv(folder/'predictions.csv')
                self.assertEqual(set(pred.row_position),set(json.loads(split.read_text())['validation_row_positions']))


if __name__=='__main__':unittest.main()
