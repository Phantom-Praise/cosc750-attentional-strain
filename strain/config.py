"""Shared dataset contract and frozen baseline definitions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]

FEATURES = ["daily_short_form_hours", "avg_clip_duration_sec", "swipes_per_minute",
            "scroll_sessions_day", "bedtime_scroll_min", "total_screen_time_hours"]

LABELS = ["Normal / Intact Focus", "Mild Attentional Strain",
          "Severe Focus Fragmentation", "Acute Brain Rot / Executive Burnout"]

SHORT_LABELS = ["Normal", "Mild", "Severe", "Acute"]

SEEDS = [42, 43, 44, 45, 46]

DATA_HASH = "2f0c2870adb1ccf7c3e3b9a35eb68e71c912868fd91068d3f81be7ba3a0b49c2"

SCORER = make_scorer(f1_score, labels=LABELS, average="macro", zero_division=0, pos_label=None)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def load_data(path):
    if sha256(path) != DATA_HASH:
        raise ValueError("Dataset hash differs from the prespecified dataset.")
    data = pd.read_csv(path)
    required = FEATURES + ["participant_id", "cognitive_decay_tier"]
    if not set(required).issubset(data.columns):
        raise ValueError("Required columns are missing.")
    if data.shape != (9000, 19) or data.isna().any().any():
        raise ValueError("Unexpected dataset shape or missing values.")
    if not data.participant_id.is_unique:
        raise ValueError("Participant identifiers must be unique for row-level splitting.")
    if set(data.cognitive_decay_tier) != set(LABELS):
        raise ValueError("Unexpected target labels.")
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Primary predictors must be finite numeric values.")
    return data

def models(seed):
    return {
        "majority": (DummyClassifier(strategy="most_frequent"), None),
        "logistic_regression": (
            make_pipeline(StandardScaler(), LogisticRegression(
                solver="lbfgs", max_iter=5000, random_state=seed)),
            {"logisticregression__C": [0.01, 0.1, 1, 10, 100, 1000],
             "logisticregression__class_weight": [None, "balanced"]}),
        "rbf_svm": (
            make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False,
                                              cache_size=256, random_state=seed)),
            {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.1],
             "svc__class_weight": [None, "balanced"]}),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
            {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 5],
             "class_weight": [None, "balanced"]}),
    }
