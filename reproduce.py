"""Portable Stage 3 reproduction: all five baselines, optionally the proposed method."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from course_models import HistogramNaiveBayes
from run_baselines import ROOT, DATA_HASH, FEATURES, LABELS, SEEDS, SCORER, load_data, models, sha256, write_json

DATA_FILE = "short_form_brain_rot_cognitive_decay.csv"
DATA_URL = ("https://www.kaggle.com/api/v1/datasets/download/samartalwar/"
            "brain-rot-short-form-video-and-attention-span?datasetVersionNumber=1")
MODEL_ORDER = ["majority", "histogram_nb", "logistic_regression", "random_forest", "rbf_svm"]
MODEL_NAMES = {"majority": "Majority reference", "histogram_nb": "Histogram Naive Bayes",
               "logistic_regression": "Logistic regression", "random_forest": "Random forest",
               "rbf_svm": "RBF-SVM", "calibrated_svm": "Calibrated SVM + decision rule"}


def obtain_data(path, download=False):
    """Download only on request; verify bytes before writing, without extracting paths."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Dataset not found: {path}. Put {DATA_FILE} there, use --data PATH, "
                                    "or pass --download-data to retrieve Kaggle version 1.")
        print("Downloading Kaggle dataset version 1 (dataset license details: DATASET.md).", flush=True)
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "COSC750-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as zipped:
            names = [n for n in zipped.namelist() if Path(n).name == DATA_FILE]
            if len(names) != 1:
                raise ValueError("Expected CSV was not uniquely identified in the downloaded archive.")
            payload = zipped.read(names[0])
        if hashlib.sha256(payload).hexdigest() != DATA_HASH:
            raise ValueError("Downloaded CSV does not match the study checksum; nothing was written.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return load_data(path)


def baseline_definitions(seed):
    definitions = models(seed)
    definitions["histogram_nb"] = (HistogramNaiveBayes(),
        {"n_bins": [3, 5, 10, 15, 20, 30], "alpha": [.5, 1.]})
    return {name: definitions[name] for name in MODEL_ORDER}


def proposed_candidates():
    return list(ParameterGrid({"C": [1, 10, 100], "gamma": ["scale", .03],
                               "cutoff": [None, .2, .35, .5]}))


def evaluate(truth, prediction):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=LABELS, zero_division=0)
    cm = confusion_matrix(truth, prediction, labels=LABELS)
    overall = {"macro_f1": float(np.mean(f1)), "accuracy": float(accuracy_score(truth, prediction)),
               "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
               "acute_precision": float(precision[-1]), "acute_recall": float(recall[-1])}
    class_rows = []
    for i, label in enumerate(LABELS):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        class_rows.append({"class": label, "support": int(support[i]), "precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i]), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return overall, class_rows, cm.tolist()


def fit_model(estimator, grid, X, y, folds, jobs):
    start = time.perf_counter()
    if grid is None:
        estimator.fit(X, y)
        elapsed = time.perf_counter() - start
        return estimator, {}, None, None, elapsed, elapsed
    search = GridSearchCV(estimator, grid, scoring=SCORER, cv=folds, n_jobs=jobs,
                          refit=True, error_score="raise")
    search.fit(X, y)
    return (search.best_estimator_, search.best_params_, float(search.best_score_),
            pd.DataFrame(search.cv_results_), time.perf_counter() - start, float(search.refit_time_))


def save_evaluation(folder, data, indices, prediction, fitted_details):
    folder.mkdir(parents=True, exist_ok=True)
    truth = data.iloc[indices].cognitive_decay_tier
    metrics, class_metrics, matrix = evaluate(truth, prediction)
    pd.DataFrame({"row_position": indices, "participant_id": data.iloc[indices].participant_id.to_numpy(),
        "true_label": truth.to_numpy(), "predicted_label": prediction}).to_csv(folder / "predictions.csv", index=False)
    write_json(folder / "metrics.json", {**metrics, **fitted_details, "confusion_matrix": matrix})
    return metrics, class_metrics


def summarize(rows, out, stem, note):
    pd.DataFrame(rows).to_csv(out / f"{stem}_per_run.csv", index=False)
    measures = ["macro_f1", "accuracy", "balanced_accuracy", "acute_precision", "acute_recall",
                "search_seconds", "refit_seconds"]
    summary = pd.DataFrame(rows).groupby("model", sort=False)[measures].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(out / f"{stem}_table.csv")
    lines = [f"# {stem.replace('_', ' ').title()}", "", note, "",
             "| Model | Macro-F1 | Accuracy | Balanced accuracy | Acute precision | Acute recall |",
             "|---|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{tabular}{lccccc}", r"\toprule",
           r"Method & Macro-F1 & Accuracy & Bal. acc. & Acute prec. & Acute recall \\", r"\midrule"]
    for name, row in summary.iterrows():
        cells = [f"{row[m+'_mean']:.4f} ± {row[m+'_std']:.4f}" for m in measures[:5]]
        lines.append("| " + " | ".join([MODEL_NAMES[name], *cells]) + " |")
        tex_cells = [f"${row[m+'_mean']:.4f} \\pm {row[m+'_std']:.4f}$" for m in measures[:5]]
        tex.append(" & ".join([MODEL_NAMES[name], *tex_cells]) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out / f"{stem}_table.md").write_text("\n".join(lines) + "\n")
    (out / f"{stem}_table.tex").write_text("\n".join(tex) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = np.arange(len(summary))
    ax.bar(x, summary.macro_f1_mean, yerr=summary.macro_f1_std, capsize=4, color="#2563eb")
    ax.set_xticks(x, [MODEL_NAMES[n] for n in summary.index], rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1 (mean ± sample SD)")
    ax.set_title("Five held-out splits" if stem == "baseline" else "Exploratory nested validation")
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.savefig(out / f"{stem}_comparison.png", dpi=180)
    fig.savefig(out / f"{stem}_comparison.svg")
    plt.close(fig)
    print("\n" + "\n".join(lines), flush=True)


def run_baseline_study(data, out, jobs):
    rows, class_rows = [], []
    for seed in SEEDS:
        train, test = train_test_split(np.arange(len(data)), test_size=.2,
            stratify=data.cognitive_decay_tier, random_state=seed)
        X, y = data.iloc[train][FEATURES], data.iloc[train].cognitive_decay_tier
        folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
        seed_dir = out / "baseline" / str(seed); seed_dir.mkdir(parents=True)
        write_json(seed_dir / "splits.json", {"train_row_positions": train.tolist(),
            "test_row_positions": test.tolist(), "cv_row_positions": [
                {"train": train[a].tolist(), "validation": train[b].tolist()} for a, b in folds]})
        for name, (estimator, grid) in baseline_definitions(seed).items():
            print(f"Baseline seed {seed} | {name}", flush=True)
            fitted, params, score, cv_results, elapsed, refit = fit_model(estimator, grid, X, y, folds, jobs)
            folder = seed_dir / name
            metrics, classes = save_evaluation(folder, data, test, fitted.predict(data.iloc[test][FEATURES]),
                {"best_params": params, "best_cv_macro_f1": score, "search_seconds": elapsed,
                 "refit_seconds": refit, "evaluation": "held_out_test"})
            if cv_results is not None: cv_results.to_csv(folder / "cv_results.csv", index=False)
            rows.append({"seed": seed, "model": name, **metrics, "search_seconds": elapsed, "refit_seconds": refit})
            class_rows.extend({"seed": seed, "model": name, **r} for r in classes)
            pd.DataFrame(rows).to_csv(out / "baseline_per_run.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "baseline_per_class.csv", index=False)
    summarize(rows, out, "baseline", "Five shared stratified 80/20 holdouts; mean ± sample SD. "
        "All learned models use the same six features and 12 training-only CV settings. "
        "Holdouts overlap; SD is not a confidence interval. Histogram NB was introduced "
        "after original baseline results were viewed. Synthetic-data results, not clinical validation.")


def run_proposed_study(data, out, jobs):
    from research_models import CalibratedAcuteSVM
    split = json.loads((out / "baseline/42/splits.json").read_text())
    training_rows = np.array(split["train_row_positions"])
    outer = list(StratifiedKFold(3, shuffle=True, random_state=2026).split(
        data.iloc[training_rows][FEATURES], data.iloc[training_rows].cognitive_decay_tier))
    configurations = proposed_candidates()
    write_json(out / "proposed_candidates.json", configurations)
    grid = [{k: [v] for k, v in p.items()} for p in configurations]
    rows, class_rows = [], []
    for fold, (a, b) in enumerate(outer):
        train, validation = training_rows[a], training_rows[b]
        X, y = data.iloc[train][FEATURES], data.iloc[train].cognitive_decay_tier
        inner = list(StratifiedKFold(3, shuffle=True, random_state=2026+fold).split(X, y))
        folder = out / "proposed_validation" / f"fold_{fold}"; folder.mkdir(parents=True)
        write_json(folder / "splits.json", {"train_row_positions": train.tolist(),
            "validation_row_positions": validation.tolist(), "inner_folds": [
                {"train": train[c].tolist(), "validation": train[d].tolist()} for c, d in inner]})
        reference, reference_grid = models(42)["rbf_svm"]
        for name, estimator, settings in [("rbf_svm", reference, reference_grid),
                ("calibrated_svm", CalibratedAcuteSVM(), grid)]:
            print(f"Proposed validation fold {fold+1}/3 | {name}", flush=True)
            fitted, params, score, cv, elapsed, refit = fit_model(estimator, settings, X, y, inner, jobs)
            metrics, classes = save_evaluation(folder/name, data, validation,
                fitted.predict(data.iloc[validation][FEATURES]), {"best_params": params,
                "best_cv_macro_f1": score, "search_seconds": elapsed, "refit_seconds": refit,
                "evaluation": "exploratory_outer_validation"})
            cv.to_csv(folder/name/"cv_results.csv", index=False)
            rows.append({"fold": fold, "model": name, **metrics, "search_seconds": elapsed, "refit_seconds": refit})
            class_rows.extend({"fold": fold, "model": name, **r} for r in classes)
    pd.DataFrame(class_rows).to_csv(out / "proposed_validation_per_class.csv", index=False)
    summarize(rows, out, "proposed_validation", "Three nested outer validation folds inside seed 42's "
        "7,200 training rows. Original test rows are excluded. Mean ± fold SD, not confidence intervals. "
        "Original SVM has 12 settings; calibrated search has 24 plus calibration fits. "
        "Exploratory findings; do not combine these scores with the five-holdout baseline table.")


def hardware():
    result = {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        for key, name in [("cpu", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")]:
            r = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
            if r.returncode == 0: result[key] = r.stdout.strip()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/DATA_FILE)
    parser.add_argument("--download-data", action="store_true", help="Retrieve verified Kaggle version 1 if CSV is absent")
    parser.add_argument("--output", type=Path, default=ROOT/"results/submission")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--proposed", action="store_true", help="Also reproduce the separate nested proposed-method comparison")
    args = parser.parse_args(argv)
    if args.jobs < 1: parser.error("--jobs must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Choose --output with a new path; earlier evidence is preserved.")
    data = obtain_data(args.data, args.download_data)
    args.output.mkdir(parents=True, exist_ok=True)
    files = ["reproduce.py", "run_baselines.py", "course_models.py", "EXPERIMENT_PLAN.md", "COURSE_EXTENSION_PLAN.md"]
    if args.proposed: files.extend(["research_models.py", "RESEARCH_EXPERIMENT_PLAN.md"])
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256(args.data), "features": FEATURES, "labels": LABELS, "seeds": SEEDS,
        "source_sha256": {p: sha256(ROOT/p) for p in files}, "hardware": hardware(), "python": sys.version,
        "versions": {p: importlib.metadata.version(p) for p in
            ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]},
        "jobs": args.jobs, "native_threads": 1, "baseline_test_fraction": .2, "baseline_cv_folds": 3,
        "baseline_search_budget": 12, "includes_proposed_validation": args.proposed,
        "command": sys.argv, "dataset_version": 1, "dataset_is_synthetic": True}
    write_json(args.output/"manifest.json", manifest)
    start = time.perf_counter()
    warnings.simplefilter("error", ConvergenceWarning)
    try:
        with threadpool_limits(limits=1), parallel_backend("threading"):
            run_baseline_study(data, args.output, args.jobs)
            if args.proposed: run_proposed_study(data, args.output, args.jobs)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc)); raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter()-start
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output/"manifest.json", manifest)
    print(f"\nComplete. Tables and evidence: {args.output.resolve()}", flush=True)


if __name__ == "__main__": main()
