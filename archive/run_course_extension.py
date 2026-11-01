"""Add lecture-based histogram NB and diagnostics without replacing baseline evidence."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import parallel_backend
from scipy.stats import norm
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from course_models import HistogramNaiveBayes
from run_baselines import ROOT, FEATURES, LABELS, SHORT_LABELS, SEEDS, SCORER, load_data, models, sha256, write_json


def wilson(successes, total):
    z = norm.ppf(.975)
    p = successes / total
    denominator = 1 + z*z/total
    midpoint = (p + z*z/(2*total))/denominator
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return float(midpoint-radius), float(midpoint+radius)


def plot_results(out, summary, curves):
    names = list(summary.index)
    display = ["Majority", "Logistic", "RBF-SVM", "Forest", "Histogram NB"]
    colors = ["#94a3b8", "#2563eb", "#0d9488", "#c27a16", "#9333ea"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].bar(range(5), summary.macro_f1_mean, yerr=summary.macro_f1_std,
                color=colors, capsize=4)
    axes[0].set_xticks(range(5), display, rotation=30, ha="right")
    axes[0].set_ylim(0,1)
    axes[0].set_title("Macro-F1: five splits, mean ± SD")
    for name, title, color in zip(names, display, colors):
        curve = curves[name]
        axes[1].plot(curve["fpr"],curve["tpr"],label=title,color=color)
        axes[2].step(curve["recall"],curve["precision"],where="post",label=title,color=color)
    axes[1].plot([0,1],[0,1],"k--",linewidth=1)
    axes[2].axhline(21/1800,color="black",linestyle="--",linewidth=1,label="Acute prevalence")
    axes[1].set(xlabel="False-positive rate",ylabel="True-positive rate",
                title="Acute vs rest ROC: seed 42",xlim=(0,1),ylim=(0,1.02))
    axes[2].set(xlabel="Recall",ylabel="Precision",
                title="Acute vs rest PR: seed 42",xlim=(0,1),ylim=(0,1.02))
    axes[2].legend(fontsize=8,loc="lower left")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Lecture-driven extension · same six usage features · exploratory comparison")
    fig.savefig(out / "course_comparison.png",dpi=170)
    fig.savefig(out / "course_comparison.svg")
    plt.close(fig)


def run(output):
    baseline = ROOT / "results/baselines"
    manifest = json.loads((baseline/"manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["script_sha256"] != sha256(ROOT/"run_baselines.py"):
        raise ValueError("Complete baseline evidence and unchanged baseline code are required.")
    if manifest["protocol_sha256"] != sha256(ROOT/"EXPERIMENT_PLAN.md"):
        raise ValueError("Original protocol changed.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory.")
    data = load_data(ROOT/"short_form_brain_rot_cognitive_decay.csv")
    output.mkdir(parents=True,exist_ok=True)
    status = {"status":"running", "baseline_manifest_sha256":sha256(baseline/"manifest.json"),
              "extension_plan_sha256":sha256(ROOT/"COURSE_EXTENSION_PLAN.md"),
              "extension_script_sha256":sha256(__file__), "model_script_sha256":sha256(ROOT/"course_models.py"),
              "data_sha256":manifest["data_sha256"],"python":platform.python_version(),
              "versions":manifest["versions"], "exploratory_after_initial_results":True,
              "lecture_sha256":{n:sha256(Path('/Users/praiseben/Downloads')/n) for n in
                  ['lec01_introduction_math_background.pdf','classification_estimation.pdf',
                   'lec02_bayes_optimal.pdf','histograms_naive_bayes.pdf']}}
    write_json(output/"manifest.json",status)
    rows, class_rows, seed42curves = [], [], {}
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=1), parallel_backend("threading"):
            for seed in SEEDS:
                folder = output/str(seed)
                folder.mkdir()
                split = json.loads((baseline/str(seed)/"splits.json").read_text())
                train, test = np.array(split["train_row_positions"]), np.array(split["test_row_positions"])
                positions = {v:i for i,v in enumerate(train)}
                folds = [(np.array([positions[v] for v in f["train"]]),
                          np.array([positions[v] for v in f["validation"]])) for f in split["cv_row_positions"]]
                Xtr, Xte = data.iloc[train][FEATURES], data.iloc[test][FEATURES]
                ytr, yte = data.iloc[train].cognitive_decay_tier, data.iloc[test].cognitive_decay_tier
                write_json(folder/"split_reference.json",{"baseline_split_sha256":sha256(baseline/str(seed)/"splits.json")})
                if seed == 42:
                    for label, short in zip(LABELS,SHORT_LABELS):
                        Xtr.loc[ytr == label].corr().to_csv(output/f"training_correlations_{short.lower()}.csv")
                comparison = models(seed)
                comparison["histogram_nb"] = (HistogramNaiveBayes(), None)
                for name, (model, _) in comparison.items():
                    destination = folder/name
                    destination.mkdir()
                    t0 = time.perf_counter()
                    if name == "histogram_nb":
                        search = GridSearchCV(model,{"n_bins":[3,5,10,15,20,30], "alpha":[.5,1.]},
                                              scoring=SCORER,cv=folds,n_jobs=4,error_score="raise")
                        search.fit(Xtr,ytr)
                        model = search.best_estimator_
                        params = search.best_params_
                        pd.DataFrame(search.cv_results_).to_csv(destination/"cv_results.csv",index=False)
                        write_json(destination/"fitted_histograms.json",{
                            "classes":model.classes_.tolist(),"class_counts":model.class_count_.tolist(),
                            "class_priors":model.class_prior_.tolist(),"best_params":params,
                            "best_cv_macro_f1":float(search.best_score_),
                            "features":{feature:{"edges":edges.tolist(),"counts":counts.tolist(),
                                                  "densities":densities.tolist()} for feature,edges,counts,densities in
                                        zip(FEATURES,model.bin_edges_,model.bin_count_,model.bin_density_)}})
                    else:
                        params = json.loads((baseline/str(seed)/name/"metrics.json").read_text())["best_params"]
                        model.set_params(**params).fit(Xtr,ytr)
                    elapsed = time.perf_counter()-t0
                    prediction = model.predict(Xte)
                    if name != "histogram_nb":
                        saved = pd.read_csv(baseline/str(seed)/name/"predictions.csv")
                        np.testing.assert_array_equal(saved.row_position,test)
                        np.testing.assert_array_equal(saved.predicted_label,prediction)
                    # Probability outputs for NB/LR/forest; uncalibrated ranking scores for SVM.
                    score_kind = "decision_function" if name == "rbf_svm" else "predict_proba"
                    raw = getattr(model,score_kind)(Xte)
                    scores = raw[:,[list(model.classes_).index(label) for label in LABELS]]
                    predictions = pd.DataFrame({"row_position":test,"true_label":yte.to_numpy(),
                                                "predicted_label":prediction})
                    for i,short in enumerate(SHORT_LABELS):
                        predictions[f"score_{short.lower()}"] = scores[:,i]
                    predictions.to_csv(destination/"predictions_and_scores.csv",index=False)
                    cm = confusion_matrix(yte,prediction,labels=LABELS)
                    precision, recall, f1, support = precision_recall_fscore_support(
                        yte,prediction,labels=LABELS,zero_division=0)
                    aucs, aps, curves = [], [], {}
                    for i,(label,short) in enumerate(zip(LABELS,SHORT_LABELS)):
                        truth = (yte.to_numpy()==label).astype(int)
                        tp = int(cm[i,i]); fn = int(cm[i,:].sum()-tp)
                        fp = int(cm[:,i].sum()-tp); tn = int(cm.sum()-tp-fn-fp)
                        auc = float(roc_auc_score(truth,scores[:,i]))
                        ap = float(average_precision_score(truth,scores[:,i]))
                        aucs.append(auc);aps.append(ap)
                        lo,hi = wilson(tp,tp+fn)
                        class_rows.append({"seed":seed,"model":name,"class":label,
                            "support":int(support[i]),"precision":float(precision[i]),"recall":float(recall[i]),
                            "f1":float(f1[i]),"specificity":tn/(tn+fp),"false_positive_rate":fp/(tn+fp),
                            "roc_auc":auc,"average_precision":ap,"recall_wilson95_low":lo,
                            "recall_wilson95_high":hi,"tp":tp,"fn":fn,"fp":fp,"tn":tn})
                        fpr,tpr,_ = roc_curve(truth,scores[:,i])
                        p,r,_ = precision_recall_curve(truth,scores[:,i])
                        curves[short] = {"fpr":fpr.tolist(),"tpr":tpr.tolist(),
                                         "precision":p.tolist(),"recall":r.tolist()}
                    row = {"seed":seed,"model":name,
                        "macro_f1":float(f1_score(yte,prediction,labels=LABELS,average="macro",zero_division=0)),
                        "balanced_accuracy":float(balanced_accuracy_score(yte,prediction)),
                        "accuracy":float(accuracy_score(yte,prediction)),
                        "empirical_zero_one_risk":float(np.mean(yte.to_numpy()!=prediction)),
                        "macro_precision":float(np.mean(precision)),"recall_acute":float(recall[-1]),
                        "macro_ovr_roc_auc":float(np.mean(aucs)),"acute_roc_auc":aucs[-1],
                        "acute_average_precision":aps[-1]}
                    write_json(destination/"diagnostics.json",{**row,"score_kind":score_kind,
                        "best_params":params,"refit_or_search_seconds":elapsed,"confusion_matrix":cm.tolist(),
                        "outside_training_range_counts":{
                            f:int(((Xte[f]<Xtr[f].min())|(Xte[f]>Xtr[f].max())).sum()) for f in FEATURES}})
                    write_json(destination/"curves.json",curves)
                    rows.append(row)
                    if seed==42:
                        seed42curves[name]=curves["Acute"]
                    print(f"Seed {seed} | {name}: macro-F1={row['macro_f1']:.4f}; "
                          f"acute recall={row['recall_acute']:.4f}",flush=True)
        table = pd.DataFrame(rows)
        table.to_csv(output/"per_seed_metrics.csv",index=False)
        pd.DataFrame(class_rows).to_csv(output/"per_class_metrics.csv",index=False)
        summary = table.groupby("model",sort=False).agg({c:["mean","std"] for c in table.columns if c not in ["seed","model"]})
        summary.columns = [f"{metric}_{stat}" for metric,stat in summary.columns]
        summary.to_csv(output/"summary.csv")
        lines = ["# Course-aligned exploratory comparison", "",
                 "Same saved splits, features, and inner folds as the original comparison. "
                 "Histogram NB was added after reading the lectures and viewing original results.", "",
                 "| Model | Macro-F1 (mean ± SD) | Acute recall (mean ± SD) | Acute AP (mean ± SD) |",
                 "|---|---:|---:|---:|"]
        for name,row in summary.iterrows():
            values = [f"{row[m+'_mean']:.4f} ± {row[m+'_std']:.4f}" for m in
                      ["macro_f1","recall_acute","acute_average_precision"]]
            lines.append("| "+" | ".join([name]+values)+" |")
        lines += ["", "AP is average precision; a constant-score acute-class reference has AP equal "
                  "to prevalence, 21/1800 = 0.0117. SD is sample standard deviation over overlapping "
                  "holdouts, not a confidence interval. The ROC/PR figure uses seed 42 only. "
                  "No threshold was selected from the test curves.", "",
                  "The original models' recomputed labels match their saved predictions exactly. "
                  "SVM outputs are uncalibrated decision scores. Histogram NB's normalized model "
                  "posteriors are not evidence of real-world or clinical probability calibration."]
        (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
        plot_results(output,summary,seed42curves)
        status["status"]="complete"
        print("\n"+"\n".join(lines),flush=True)
    except BaseException as exc:
        status.update(status="failed",error=repr(exc))
        raise
    finally:
        status["elapsed_seconds"]=time.perf_counter()-start
        write_json(output/"manifest.json",status)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/course_extension")
    run(parser.parse_args().output)
