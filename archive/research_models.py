"""Research-inspired estimators. Upstream algorithms are credited in the review."""
from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import make_pipeline as make_sampling_pipeline
from mord.threshold_based import obj_margin, grad_margin, threshold_predict, threshold_proba

from run_baselines import LABELS


def weight_choices():
    return [None,*[{label:factor if label==LABELS[-1] else 1 for label in LABELS}
                    for factor in [2,4]]]


class OrdinalLogistic(ClassifierMixin, BaseEstimator):
    """mord immediate-threshold objective with explicit ordering and checked solver.

    Uses mord 0.7 routines by Fabian Pedregosa and contributors (BSD license).
    The modern wrapper, scaling, label mapping, and solver checks are project code.
    """
    def __init__(self, alpha=1., class_weight=None):
        self.alpha=alpha
        self.class_weight=class_weight

    def fit(self,X,y):
        X,y=check_X_y(X,y,dtype=float)
        if set(y)!=set(LABELS):
            raise ValueError("All four explicitly ordered classes are required.")
        self.scaler_=StandardScaler().fit(X)
        Z=self.scaler_.transform(X)
        encoded=np.array([LABELS.index(label) for label in y])
        sample_weight=compute_sample_weight(self.class_weight,y)
        c,d=len(LABELS),Z.shape[1]
        L=np.tril(np.ones((c-1,c-1)))
        ranks=np.arange(c)[:,None];cutpoints=np.arange(c-1)[None,:]
        loss_fd=((ranks==cutpoints)|(ranks==cutpoints+1)).astype(float)
        initial=np.r_[np.zeros(d),0.,np.ones(c-2)]
        result=minimize(obj_margin,initial,method="L-BFGS-B",jac=grad_margin,
            bounds=[(None,None)]*(d+1)+[(1e-6,None)]*(c-2),
            args=(Z,encoded,self.alpha,c,loss_fd,L,sample_weight),
            options={"maxiter":2000,"ftol":1e-9,"gtol":1e-5})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Ordinal optimization failed: {result.message}")
        self.coef_=result.x[:d]
        self.thresholds_=L@result.x[d:]
        self.classes_=np.array(LABELS)
        self.n_features_in_=d
        self.optimization_={"success":bool(result.success),"iterations":int(result.nit),
                            "objective":float(result.fun),"message":str(result.message)}
        return self

    def predict(self,X):
        check_is_fitted(self,"thresholds_")
        X=check_array(X,dtype=float)
        return self.classes_[threshold_predict(self.scaler_.transform(X),self.coef_,self.thresholds_)]

    def predict_proba(self,X):
        check_is_fitted(self,"thresholds_")
        X=check_array(X,dtype=float)
        return threshold_proba(self.scaler_.transform(X),self.coef_,self.thresholds_)


class PreprocessedSVM(ClassifierMixin, BaseEstimator):
    def __init__(self, transform="standard", C=1., gamma="scale"):
        self.transform=transform;self.C=C;self.gamma=gamma

    def fit(self,X,y):
        transformers={"standard":StandardScaler(),"robust":RobustScaler(),
                      "quantile":QuantileTransformer(n_quantiles=min(100,len(X)),
                          output_distribution="normal",random_state=2026)}
        self.model_=make_pipeline(transformers[self.transform],SVC(C=self.C,gamma=self.gamma))
        self.model_.fit(X,y)
        self.classes_=self.model_.classes_
        self.n_features_in_=self.model_.n_features_in_
        return self

    def predict(self,X): return self.model_.predict(X)


def acute_counts(y,multiplier):
    return {LABELS[-1]:int(np.sum(np.asarray(y)==LABELS[-1])*multiplier)}


class OversampledSVM(ClassifierMixin, BaseEstimator):
    def __init__(self, sampling="random", multiplier=2, C=1.):
        self.sampling=sampling;self.multiplier=multiplier;self.C=C

    def fit(self,X,y):
        strategy=partial(acute_counts,multiplier=self.multiplier)
        sampler=(RandomOverSampler(sampling_strategy=strategy,random_state=2026)
                 if self.sampling=="random" else SMOTE(sampling_strategy=strategy,
                                                       k_neighbors=3,random_state=2026))
        self.model_=make_sampling_pipeline(StandardScaler(),sampler,SVC(C=self.C,gamma="scale"))
        self.model_.fit(X,y)
        self.classes_=self.model_.classes_
        self.n_features_in_=self.model_.n_features_in_
        self.requested_resampled_acute_count_=acute_counts(y,self.multiplier)[LABELS[-1]]
        return self

    def predict(self,X): return self.model_.predict(X)


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


class EarlyStoppedBoosting(ClassifierMixin, BaseEstimator):
    def __init__(self,learning_rate=.03,max_depth=3,max_leaf_nodes=15,
                 min_samples_leaf=20,l2_regularization=1.,class_weight=None):
        self.learning_rate=learning_rate;self.max_depth=max_depth
        self.max_leaf_nodes=max_leaf_nodes;self.min_samples_leaf=min_samples_leaf
        self.l2_regularization=l2_regularization;self.class_weight=class_weight

    def fit(self,X,y):
        self.model_=HistGradientBoostingClassifier(learning_rate=self.learning_rate,
            max_depth=self.max_depth,max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,l2_regularization=self.l2_regularization,
            max_iter=1000,early_stopping=True,validation_fraction=.2,n_iter_no_change=20,
            scoring="loss",random_state=2026,class_weight=None)
        self.model_.fit(X,y,sample_weight=compute_sample_weight(self.class_weight,y))
        self.classes_=self.model_.classes_
        self.n_features_in_=self.model_.n_features_in_
        self.iterations_=int(self.model_.n_iter_)
        return self

    def predict(self,X): return self.model_.predict(X)
    def predict_proba(self,X): return self.model_.predict_proba(X)
