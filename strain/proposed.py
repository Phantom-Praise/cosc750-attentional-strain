"""Calibrated RBF-SVM with validation-selected four-class decision rule."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .config import LABELS

class CalibratedAcuteSVM(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1., gamma="scale", cutoff=None):
        self.C=C;self.gamma=gamma;self.cutoff=cutoff

    def fit(self,X,y):
        # The scaler is inside the calibration estimator, so it is refitted on
        # each calibration-training subset, rather than using its validation rows.
        pipeline=make_pipeline(StandardScaler(),SVC(C=self.C,gamma=self.gamma))
        self.model_=CalibratedClassifierCV(pipeline,method="sigmoid",ensemble=True,
            cv=StratifiedKFold(3,shuffle=True,random_state=2026),n_jobs=1).fit(X,y)
        self.classes_=self.model_.classes_
        self.n_features_in_=self.model_.n_features_in_
        return self

    def predict_proba(self,X): return self.model_.predict_proba(X)

    def predict(self,X):
        p=self.predict_proba(X)
        if self.cutoff is None: return self.classes_[p.argmax(axis=1)]
        acute=list(self.classes_).index(LABELS[-1])
        alternatives=p.copy();alternatives[:,acute]=-np.inf
        indices=alternatives.argmax(axis=1)
        indices[p[:,acute]>=self.cutoff]=acute
        return self.classes_[indices]
