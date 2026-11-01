"""Exploratory nested validation of expanded SVM and histogram gradient boosting."""
import argparse
import json
import os
import time
from pathlib import Path

# OpenMP thread limits must be set before importing numerical libraries so each
# joblib worker thread inherits the intended limit, including HGB's native loops.
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from threadpoolctl import threadpool_limits

from run_baselines import ROOT, LABELS, SHORT_LABELS, FEATURES, SCORER, load_data, models, sha256, write_json


from tuning_models import WeightedBoosting


def candidates():
    weights = [None,"balanced",{label:2 if label==LABELS[-1] else 1 for label in LABELS},
               {label:4 if label==LABELS[-1] else 1 for label in LABELS}]
    original = list(ParameterGrid(models(42)["rbf_svm"][1]))
    broader = list(ParameterGrid({"svc__C":[.1,.3,1,3,10,30,100,300],
        "svc__gamma":["scale",.001,.003,.01,.03,.1,.3,1],"svc__class_weight":weights}))
    # A local seed makes the sampled candidate list independent of model outcomes.
    extra = [p for p in broader if p not in original]
    rng = np.random.default_rng(2026)
    expanded = original + [extra[i] for i in rng.choice(len(extra),24,replace=False)]
    boosting = list(ParameterGrid({"learning_rate":[.03,.06,.1,.2],"max_iter":[100,200,300],
        "max_leaf_nodes":[7,15,31],"min_samples_leaf":[10,20,40],
        "l2_regularization":[0,.1,1,10],"class_weight":weights}))
    rng = np.random.default_rng(2026)
    return {"original_svm":original,"expanded_svm":expanded,
            "gradient_boosting":[boosting[i] for i in rng.choice(len(boosting),36,replace=False)]}


def estimator(name):
    if name in ("original_svm","expanded_svm"):
        return models(42)["rbf_svm"][0]
    return WeightedBoosting()


def search(name, configurations, X, y, folds, jobs):
    # Singleton dictionaries make GridSearchCV evaluate exactly the frozen candidates.
    grid = [{k:[v] for k,v in config.items()} for config in configurations]
    result = GridSearchCV(estimator(name),grid,scoring=SCORER,cv=folds,n_jobs=jobs,
                          refit=True,error_score="raise")
    start=time.perf_counter()
    result.fit(X,y)
    return result,time.perf_counter()-start


def metrics(y,prediction):
    p,r,f,_=precision_recall_fscore_support(y,prediction,labels=LABELS,zero_division=0)
    return {"macro_f1":float(np.mean(f)),"accuracy":float(accuracy_score(y,prediction)),
            "balanced_accuracy":float(balanced_accuracy_score(y,prediction)),
            "macro_precision":float(np.mean(p)),"acute_precision":float(p[-1]),
            "acute_recall":float(r[-1])}


def run(out,jobs):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory.")
    baseline=ROOT/"results/baselines"
    old=json.loads((baseline/"manifest.json").read_text())
    if old["status"]!="complete" or old["script_sha256"]!=sha256(ROOT/"run_baselines.py"):
        raise ValueError("Original baseline evidence or code changed.")
    data=load_data(ROOT/"short_form_brain_rot_cognitive_decay.csv")
    split=json.loads((baseline/"42/splits.json").read_text())
    train=np.array(split["train_row_positions"])
    excluded=set(split["test_row_positions"])
    assert not set(train)&excluded
    X=data.iloc[train][FEATURES].reset_index(drop=True)
    y=data.iloc[train].cognitive_decay_tier.reset_index(drop=True)
    configs=candidates()
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"candidates.json",configs)
    manifest={"status":"running","exploratory":True,"uses_original_seed42_test":False,
        "script_sha256":sha256(__file__),"model_script_sha256":sha256(ROOT/"tuning_models.py"),"plan_sha256":sha256(ROOT/"TUNING_PLAN.md"),
        "candidate_sha256":sha256(out/"candidates.json"),"versions":old["versions"],
        "data_sha256":old["data_sha256"],"baseline_split_sha256":sha256(baseline/"42/splits.json"),
        "outer_folds":3,"inner_folds":3,"jobs":jobs,"native_threads":1}
    write_json(out/"manifest.json",manifest)
    start=time.perf_counter()
    records=[]
    predictions={name:np.empty(len(train),dtype=object) for name in configs}
    assigned=np.zeros(len(train),dtype=int)
    try:
        with threadpool_limits(limits=1),parallel_backend("threading"):
            outer=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(X,y))
            for fold,(a,b) in enumerate(outer):
                fold_dir=out/f"fold_{fold}"
                fold_dir.mkdir()
                inner=list(StratifiedKFold(3,shuffle=True,random_state=2026+fold).split(X.iloc[a],y.iloc[a]))
                write_json(fold_dir/"splits.json",{
                    "train_row_positions":train[a].tolist(),"validation_row_positions":train[b].tolist(),
                    "inner_folds":[{"train":train[a[c]].tolist(),"validation":train[a[d]].tolist()}
                                   for c,d in inner]})
                assigned[b]+=1
                for name,parameters in configs.items():
                    folder=fold_dir/name
                    folder.mkdir()
                    print(f"Outer fold {fold+1}/3 | {name}: {len(parameters)} candidates × 3 folds",flush=True)
                    result,elapsed=search(name,parameters,X.iloc[a],y.iloc[a],inner,jobs)
                    prediction=result.predict(X.iloc[b])
                    predictions[name][b]=prediction
                    row={"fold":fold,"model":name,**metrics(y.iloc[b],prediction),"search_seconds":elapsed}
                    records.append(row)
                    pd.DataFrame(records).to_csv(out/"fold_metrics.csv",index=False)
                    pd.DataFrame(result.cv_results_).to_csv(folder/"cv_results.csv",index=False)
                    write_json(folder/"selection.json",{"best_params":result.best_params_,
                        "inner_cv_macro_f1":float(result.best_score_),"outer_metrics":row})
                    print(f"  macro-F1={row['macro_f1']:.4f}; acute precision={row['acute_precision']:.4f}; "
                          f"acute recall={row['acute_recall']:.4f}; {elapsed:.1f}s",flush=True)
            assert (assigned==1).all()
            table=pd.DataFrame(records)
            columns=["macro_f1","accuracy","balanced_accuracy","macro_precision",
                     "acute_precision","acute_recall","search_seconds"]
            summary=table.groupby("model",sort=False)[columns].agg(["mean","std"])
            summary.columns=[f"{metric}_{stat}" for metric,stat in summary.columns]
            summary.to_csv(out/"summary.csv")
            pooled=[]
            for name,prediction in predictions.items():
                pd.DataFrame({"row_position":train,"true_label":y,"predicted_label":prediction}
                             ).to_csv(out/f"{name}_oof_predictions.csv",index=False)
                p,r,f,s=precision_recall_fscore_support(y,prediction,labels=LABELS,zero_division=0)
                cm=confusion_matrix(y,prediction,labels=LABELS)
                write_json(out/f"{name}_pooled_metrics.json",{**metrics(y,prediction),"confusion_matrix":cm.tolist()})
                for i,label in enumerate(LABELS):
                    tp=int(cm[i,i]);fp=int(cm[:,i].sum()-tp);fn=int(cm[i,:].sum()-tp)
                    pooled.append({"model":name,"class":label,"support":int(s[i]),
                        "precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),
                        "tp":tp,"fp":fp,"fn":fn,"tn":int(cm.sum()-tp-fp-fn)})
            pd.DataFrame(pooled).to_csv(out/"pooled_per_class_metrics.csv",index=False)
            new_names=["expanded_svm","gradient_boosting"]
            chosen=max(new_names,key=lambda name:summary.loc[name,"macro_f1_mean"])
            print(f"Refitting exploratory choice on 7,200 training rows: {chosen}",flush=True)
            final_folds=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(X,y))
            final,elapsed=search(chosen,configs[chosen],X,y,final_folds,jobs)
            pd.DataFrame(final.cv_results_).to_csv(out/"final_training_cv_results.csv",index=False)
            joblib.dump({"estimator":final.best_estimator_,"features":FEATURES,"labels":LABELS,
                         "training_row_positions":train,"model":chosen,"exploratory":True},out/"model.joblib")
            write_json(out/"final_model.json",{"model":chosen,"best_params":final.best_params_,
                "training_cv_macro_f1":float(final.best_score_),"search_seconds":elapsed,
                "test_evaluated":False})
            lines=["# Exploratory tuning results", "",
                "Nested three-fold CV within original seed 42's 7,200 training rows. "
                "Original test rows excluded. These are validation scores, not new test results. "
                "Earlier test results have been viewed, so this study remains exploratory.","",
                "| Search procedure | Macro-F1 (mean ± SD) | Acute precision (mean ± SD) | Acute recall (mean ± SD) |",
                "|---|---:|---:|---:|"]
            for name,row in summary.iterrows():
                cells=[f"{row[m+'_mean']:.4f} ± {row[m+'_std']:.4f}" for m in ["macro_f1","acute_precision","acute_recall"]]
                lines.append("| "+" | ".join([name]+cells)+" |")
            lines += ["", "SD is over three overlapping-training outer folds, not a confidence interval. "
                "Each validation fold contains 28 acute records; pooled out-of-fold predictions "
                "cover 7,200 unique rows, including 84 acute records.","",
                f"Exploratory model choice: {chosen}. Its fixed search was repeated on all "
                "7,200 training rows and the refitted model is saved as model.joblib. The "
                "model was not evaluated on the original test rows. Do not compare this "
                "table directly against the earlier five-holdout test averages."]
            (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
            fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
            for ax,m,title in zip(axes,["macro_f1","acute_precision","acute_recall"],
                                 ["Macro-F1","Acute precision","Acute recall"]):
                ax.bar(range(3),summary[m+"_mean"],yerr=summary[m+"_std"],capsize=4,
                       color=["#94a3b8","#0d9488","#9333ea"])
                ax.set_xticks(range(3),["Original SVM","Expanded SVM","Boosting"],rotation=20)
                ax.set_ylim(0,1.1);ax.set_title(title);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
            fig.suptitle("Exploratory nested validation · training partition only · mean ± fold SD")
            fig.savefig(out/"tuning_comparison.png",dpi=180)
            fig.savefig(out/"tuning_comparison.svg")
            plt.close(fig)
            print("\n"+"\n".join(lines),flush=True)
        manifest["status"]="complete"
    except BaseException as exc:
        manifest.update(status="failed",error=repr(exc));raise
    finally:
        manifest["elapsed_seconds"]=time.perf_counter()-start
        write_json(out/"manifest.json",manifest)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/tuning")
    parser.add_argument("--jobs",type=int,default=4)
    args=parser.parse_args()
    run(args.output,args.jobs)
