"""Fold-local sample weighting for histogram gradient boosting."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


class WeightedBoosting(ClassifierMixin, BaseEstimator):
    """Apply weights before HGB's internal label encoding, within each CV fit."""
    def __init__(self, learning_rate=.1, max_iter=100, max_leaf_nodes=31,
                 min_samples_leaf=20, l2_regularization=0, class_weight=None):
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        self.max_leaf_nodes=max_leaf_nodes
        self.min_samples_leaf=min_samples_leaf
        self.l2_regularization=l2_regularization
        self.class_weight=class_weight

    def fit(self,X,y):
        weights=compute_sample_weight(self.class_weight,y)
        self.model_=HistGradientBoostingClassifier(learning_rate=self.learning_rate,
            max_iter=self.max_iter,max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,l2_regularization=self.l2_regularization,
            early_stopping=False,random_state=2026,class_weight=None)
        self.model_.fit(X,y,sample_weight=weights)
        self.classes_=self.model_.classes_
        self.n_features_in_=self.model_.n_features_in_
        self.fit_class_weights_={str(label):float(weights[np.asarray(y)==label][0]) for label in self.classes_}
        return self

    def predict(self,X):
        return self.model_.predict(X)

    def predict_proba(self,X):
        return self.model_.predict_proba(X)

