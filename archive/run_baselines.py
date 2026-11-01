"""Reproduce the prespecified behavioral-feature baseline comparison."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, make_scorer, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
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


def summarize(out):
    scores = pd.read_csv(out / "per_seed_metrics.csv")
    metric_cols = ["macro_f1", "balanced_accuracy", "accuracy", "recall_normal",
                   "recall_mild", "recall_severe", "recall_acute", "search_seconds",
                   "refit_seconds", "predict_seconds"]
    summary = scores.groupby("model", sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(out / "summary.csv")
    lines = ["# Behavioral-feature baseline results", "",
             "Five stratified 80/20 holdouts; mean ± sample standard deviation. "
             "Hyperparameters selected by training-only three-fold CV macro-F1.", "",
             "| Model | Macro-F1 | Balanced accuracy | Accuracy | Acute recall |",
             "|---|---:|---:|---:|---:|"]
    for name, row in summary.iterrows():
        cells = [f"{row[m + '_mean']:.4f} ± {row[m + '_std']:.4f}"
                 for m in ["macro_f1", "balanced_accuracy", "accuracy", "recall_acute"]]
        lines.append("| " + " | ".join([name] + cells) + " |")
    lines += ["", "Each test set has 1,800 records, including 21 acute-class records. "
              "Repeated test sets overlap; these are not five independent datasets. "
              "Scores describe synthetic-data prediction, not clinical validity.", "",
              "Search time includes validation fits and final refit; refit time is also "
              "reported separately and must not be added to search time."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = list(summary.index)
    display_names = ["Majority", "Logistic\nregression", "RBF-SVM", "Random\nforest"]
    for ax, metric, title in zip(axes, ["macro_f1", "recall_acute"],
                                 ["Macro-F1", "Recall: acute class"]):
        ax.bar(range(len(names)), summary[metric + "_mean"],
               yerr=summary[metric + "_std"], capsize=4,
               color=["#94a3b8", "#2563eb", "#0d9488", "#c27a16"])
        ax.set_xticks(range(len(names)), display_names)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Score (mean ± sample SD)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Six usage features · five stratified test splits")
    fig.savefig(out / "baseline_comparison.png", dpi=180)
    fig.savefig(out / "baseline_comparison.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, name, title in zip(axes, names, display_names):
        matrices = [np.array(json.loads((out / str(seed) / name / "metrics.json").read_text())
                             ["confusion_matrix"]) for seed in SEEDS]
        summed = np.sum(matrices, axis=0)
        normalized = summed / summed.sum(axis=1, keepdims=True)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                        color="white" if normalized[i, j] > .55 else "black", fontsize=9)
        ax.set_xticks(range(4), SHORT_LABELS, rotation=45)
        ax.set_yticks(range(4), SHORT_LABELS)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title.replace("\n", " "))
    fig.suptitle("Row-normalized confusion matrices pooled over five overlapping test splits")
    fig.savefig(out / "confusion_matrices.png", dpi=180)
    fig.savefig(out / "confusion_matrices.svg")
    plt.close(fig)
    print("\n" + "\n".join(lines), flush=True)


def run(args):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output to preserve results.")
    data = load_data(args.data)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data.resolve()), "data_sha256": sha256(args.data),
        "script_sha256": sha256(__file__), "protocol_sha256": sha256(ROOT / "EXPERIMENT_PLAN.md"),
        "features": FEATURES, "labels": LABELS, "seeds": SEEDS, "cv_folds": 3,
        "test_fraction": .2, "search_jobs": args.jobs, "parallel_backend": "threading",
        "native_threads": 1, "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "command": sys.argv,
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]},
        "class_counts": {label: int((data.cognitive_decay_tier == label).sum()) for label in LABELS},
    }
    write_json(out / "manifest.json", manifest)
    all_metrics = []
    warnings.simplefilter("error", ConvergenceWarning)
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=1), parallel_backend("threading"):
            for seed in SEEDS:
                train, test = train_test_split(np.arange(len(data)), test_size=.2,
                                               stratify=data.cognitive_decay_tier, random_state=seed)
                assert not set(train) & set(test)
                assert len(train) + len(test) == len(data)
                seed_dir = out / str(seed)
                seed_dir.mkdir()
                X_train, X_test = data.iloc[train][FEATURES], data.iloc[test][FEATURES]
                y_train = data.iloc[train].cognitive_decay_tier
                y_test = data.iloc[test].cognitive_decay_tier
                folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X_train, y_train))
                write_json(seed_dir / "splits.json", {
                    "train_row_positions": train.tolist(), "test_row_positions": test.tolist(),
                    "cv_row_positions": [{"train": train[a].tolist(), "validation": train[b].tolist()}
                                         for a, b in folds]})
                for name, (estimator, grid) in models(seed).items():
                    model_dir = seed_dir / name
                    model_dir.mkdir()
                    print(f"Seed {seed} | {name}: fitting", flush=True)
                    t0 = time.perf_counter()
                    if grid:
                        search = GridSearchCV(estimator, grid, scoring=SCORER, cv=folds,
                                              n_jobs=args.jobs, refit=True, error_score="raise",
                                              return_train_score=False)
                        search.fit(X_train, y_train)
                        fitted = search.best_estimator_
                        search_seconds = time.perf_counter() - t0
                        refit_seconds = search.refit_time_
                        params, validation_score = search.best_params_, float(search.best_score_)
                        pd.DataFrame(search.cv_results_).to_csv(model_dir / "cv_results.csv", index=False)
                    else:
                        fitted = estimator.fit(X_train, y_train)
                        refit_seconds = time.perf_counter() - t0
                        search_seconds = refit_seconds
                        params, validation_score = {}, None
                    t1 = time.perf_counter()
                    prediction = fitted.predict(X_test)
                    predict_seconds = time.perf_counter() - t1
                    recalls = recall_score(y_test, prediction, labels=LABELS, average=None, zero_division=0)
                    row = {
                        "seed": seed, "model": name,
                        "macro_f1": float(f1_score(y_test, prediction, labels=LABELS,
                                                    average="macro", zero_division=0)),
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
                        "accuracy": float(accuracy_score(y_test, prediction)),
                        **{f"recall_{label.lower()}": float(r) for label, r in zip(SHORT_LABELS, recalls)},
                        "search_seconds": search_seconds, "refit_seconds": refit_seconds,
                        "predict_seconds": predict_seconds,
                    }
                    write_json(model_dir / "metrics.json", {
                        **row, "best_params": params, "best_cv_macro_f1": validation_score,
                        "confusion_matrix": confusion_matrix(y_test, prediction, labels=LABELS).tolist(),
                        "test_class_counts": {label: int((y_test == label).sum()) for label in LABELS},
                    })
                    pd.DataFrame({"row_position": test,
                                  "participant_id": data.iloc[test].participant_id.to_numpy(),
                                  "true_label": y_test.to_numpy(), "predicted_label": prediction}
                                 ).to_csv(model_dir / "predictions.csv", index=False)
                    all_metrics.append(row)
                    pd.DataFrame(all_metrics).to_csv(out / "per_seed_metrics.csv", index=False)
                    print(f"  macro-F1={row['macro_f1']:.4f}; acute recall={row['recall_acute']:.4f}; "
                          f"search+refit={search_seconds:.1f}s", flush=True)
        summarize(out)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - start
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "short_form_brain_rot_cognitive_decay.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "baselines")
    parser.add_argument("--jobs", type=int, default=4)
    run(parser.parse_args())
