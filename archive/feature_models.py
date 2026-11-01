"""Explicit feature sets and fold-local preprocessing for the ablation study."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from run_baselines import FEATURES

CONTEXT=["total_sleep_hours","multitasking_score"]
DEMOGRAPHICS=["age","gender","primary_platform"]
PERMITTED_INPUTS=FEATURES+CONTEXT+DEMOGRAPHICS
DERIVED=["short_form_minutes_per_session","swipe_clip_ratio","bedtime_scroll_share",
         "short_form_screen_share"]


def feature_frame(X,variant):
    if not isinstance(X,pd.DataFrame):
        raise ValueError("Named raw-input columns are required.")
    columns=FEATURES+(CONTEXT if variant=="context" else DEMOGRAPHICS if variant=="demographics_platform" else [])
    frame=X.loc[:,columns].copy()
    if variant in ("engineered","engineered_log"):
        h=frame.daily_short_form_hours.to_numpy(dtype=float)
        sessions=frame.scroll_sessions_day.to_numpy(dtype=float)
        screen=frame.total_screen_time_hours.to_numpy(dtype=float)
        if (h<=0).any() or (sessions<=0).any() or (screen<=0).any():
            raise ValueError("Derived-ratio denominators must be positive.")
        frame[DERIVED[0]]=60*h/sessions
        frame[DERIVED[1]]=frame.swipes_per_minute*frame.avg_clip_duration_sec/60
        frame[DERIVED[2]]=frame.bedtime_scroll_min/(60*h)
        frame[DERIVED[3]]=h/screen
    if variant=="engineered_log":
        if (frame[FEATURES]<0).any().any():raise ValueError("Log inputs must be nonnegative.")
        for column in FEATURES:frame["log1p_"+column]=np.log1p(frame[column])
    return frame


class FeatureSVM(ClassifierMixin,BaseEstimator):
    def __init__(self,variant="context",C=1.,gamma="scale",class_weight=None):
        self.variant=variant;self.C=C;self.gamma=gamma;self.class_weight=class_weight

    def fit(self,X,y):
        frame=feature_frame(X,self.variant)
        categorical=[c for c in ["gender","primary_platform"] if c in frame]
        numeric=[c for c in frame if c not in categorical]
        parts=[("numeric",StandardScaler(),numeric)]
        if categorical:parts.append(("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical))
        preprocessing=ColumnTransformer(parts,verbose_feature_names_out=False)
        self.model_=make_pipeline(preprocessing,SVC(C=self.C,gamma=self.gamma,class_weight=self.class_weight))
        self.model_.fit(frame,y)
        self.classes_=self.model_.classes_
        self.output_feature_names_=preprocessing.get_feature_names_out().tolist()
        self.raw_input_features_=FEATURES+(CONTEXT if self.variant=="context" else DEMOGRAPHICS if self.variant=="demographics_platform" else [])
        self.model_input_features_=frame.columns.tolist()
        return self

    def predict(self,X):return self.model_.predict(feature_frame(X,self.variant))


class InteractionLogistic(ClassifierMixin,BaseEstimator):
    def __init__(self,interactions=False,C=1.,class_weight=None):
        self.interactions=interactions;self.C=C;self.class_weight=class_weight

    def fit(self,X,y):
        frame=feature_frame(X,"base")
        steps=[StandardScaler()]
        if self.interactions:
            steps.extend([PolynomialFeatures(degree=2,interaction_only=True,include_bias=False),StandardScaler()])
        steps.append(LogisticRegression(C=self.C,class_weight=self.class_weight,solver="lbfgs",
                                        max_iter=5000,random_state=2026))
        self.model_=make_pipeline(*steps).fit(frame,y)
        self.classes_=self.model_.classes_
        self.raw_input_features_=FEATURES.copy()
        self.output_feature_names_=(self.model_.named_steps['polynomialfeatures'].get_feature_names_out(FEATURES).tolist()
                                    if self.interactions else FEATURES.copy())
        return self

    def predict(self,X):return self.model_.predict(feature_frame(X,"base"))
