import unittest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from strain.config import LABELS
from strain.proposed import CalibratedAcuteSVM


class ProposedModelTests(unittest.TestCase):

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
