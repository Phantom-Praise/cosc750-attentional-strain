"""Compare explicit feature additions on the saved training-only nested CV folds."""
import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
import argparse
import importlib.metadata
import json
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV,ParameterGrid
from threadpoolctl import threadpool_limits

from run_baselines import ROOT,LABELS,FEATURES,SCORER,load_data,sha256,write_json
from run_tuning import metrics
from feature_models import FeatureSVM,InteractionLogistic,PERMITTED_INPUTS


def configurations():
    svm=list(ParameterGrid({"C":[.1,1,10],"gamma":["scale",.1],"class_weight":[None,"balanced"]}))
    logistic=list(ParameterGrid({"C":[.01,.1,1,10,100,1000],"class_weight":[None,"balanced"]}))
    result={}
    for variant in ["context","demographics_platform","engineered","engineered_log"]:
        result["svm_"+variant]=[{**p,"variant":variant} for p in svm]
    result["logistic_linear"]=[{**p,"interactions":False} for p in logistic]
    result["logistic_interactions"]=[{**p,"interactions":True} for p in logistic]
    return result


def estimator(name):
    return InteractionLogistic() if name.startswith("logistic_") else FeatureSVM()


def rank_mae(y,pred):
    mapping={label:i for i,label in enumerate(LABELS)}
    return float(np.mean(np.abs(np.array([mapping[v] for v in y])-np.array([mapping[v] for v in pred]))))


def run(out,jobs):
    if out.exists() and any(out.iterdir()):raise ValueError("Choose an empty output directory.")
    previous=ROOT/"results/tuning"
    if json.loads((previous/"manifest.json").read_text())["status"]!="complete":
        raise ValueError("Completed nested comparison is required.")
    data=load_data(ROOT/"short_form_brain_rot_cognitive_decay.csv")
    split=json.loads((ROOT/"results/baselines/42/splits.json").read_text())
    rows=np.array(split["train_row_positions"]);excluded=set(split["test_row_positions"])
    X=data.iloc[rows][PERMITTED_INPUTS].reset_index(drop=True)
    y=data.iloc[rows].cognitive_decay_tier.reset_index(drop=True)
    positions={int(v):i for i,v in enumerate(rows)}
    configs=configurations()
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"candidates.json",configs)
    reference=pd.read_csv(previous/"original_svm_oof_predictions.csv").set_index("row_position")
    np.testing.assert_array_equal(reference.loc[rows].true_label,y)
    oof={"original_svm":reference.loc[rows].predicted_label.to_numpy()}
    for name in configs:oof[name]=np.empty(len(rows),dtype=object)
    records=[];covered=np.zeros(len(rows),dtype=int)
    manifest={"status":"running","exploratory":True,"test_evaluated":False,
        "script_sha256":sha256(__file__),"model_script_sha256":sha256(ROOT/"feature_models.py"),
        "plan_sha256":sha256(ROOT/"FEATURE_ABLATION_PLAN.md"),
        "candidate_sha256":sha256(out/"candidates.json"),
        "reference_oof_sha256":sha256(previous/"original_svm_oof_predictions.csv"),
        "versions":{n:importlib.metadata.version(n) for n in
                    ["scikit-learn","numpy","scipy","pandas","imbalanced-learn","mord"]},
        "jobs":jobs,"native_threads":1,"candidate_budget":12,"outer_folds":3,"inner_folds":3,
        "data_sha256":sha256(ROOT/"short_form_brain_rot_cognitive_decay.csv"),
        "permitted_raw_inputs":PERMITTED_INPUTS,
        "training_ratio_audit":{
            "bedtime_minutes_exceed_daily_short_form_minutes":int((X.bedtime_scroll_min>60*X.daily_short_form_hours).sum()),
            "short_form_hours_exceed_total_screen_hours":int((X.daily_short_form_hours>X.total_screen_time_hours).sum())}}
    write_json(out/"manifest.json",manifest)
    start=time.perf_counter()
    warnings.simplefilter("error",ConvergenceWarning)
    try:
        with threadpool_limits(limits=1),parallel_backend("threading"):
            for fold in range(3):
                saved=previous/f"fold_{fold}/splits.json"
                s=json.loads(saved.read_text())
                a=np.array([positions[v] for v in s["train_row_positions"]])
                b=np.array([positions[v] for v in s["validation_row_positions"]])
                assert not (set(s["train_row_positions"])|set(s["validation_row_positions"]))&excluded
                lookup={int(v):i for i,v in enumerate(s["train_row_positions"])}
                inner=[(np.array([lookup[v] for v in f["train"]]),
                        np.array([lookup[v] for v in f["validation"]])) for f in s["inner_folds"]]
                folder=out/f"fold_{fold}";folder.mkdir()
                write_json(folder/"splits.json",s)
                reference_pred=oof["original_svm"][b]
                records.append({"fold":fold,"model":"original_svm",**metrics(y.iloc[b],reference_pred),
                    "rank_mae":rank_mae(y.iloc[b],reference_pred),"search_seconds":0.,"reference_reused":True})
                covered[b]+=1
                for name,parameters in configs.items():
                    destination=folder/name;destination.mkdir()
                    print(f"Fold {fold+1}/3 | {name}: 12 candidates × 3 inner folds",flush=True)
                    grid=[{key:[value] for key,value in p.items()} for p in parameters]
                    search=GridSearchCV(estimator(name),grid,cv=inner,scoring=SCORER,
                        n_jobs=jobs,error_score="raise",refit=True)
                    t0=time.perf_counter();search.fit(X.iloc[a],y.iloc[a]);elapsed=time.perf_counter()-t0
                    pred=search.predict(X.iloc[b]);oof[name][b]=pred
                    row={"fold":fold,"model":name,**metrics(y.iloc[b],pred),
                         "rank_mae":rank_mae(y.iloc[b],pred),"search_seconds":elapsed,"reference_reused":False}
                    records.append(row)
                    details={"best_params":search.best_params_,"best_cv_macro_f1":float(search.best_score_),
                             "outer_metrics":row}
                    fitted=search.best_estimator_
                    for attribute in ["raw_input_features_","model_input_features_","output_feature_names_"]:
                        if hasattr(fitted,attribute):details[attribute]=getattr(fitted,attribute)
                    write_json(destination/"selection.json",details)
                    pd.DataFrame(search.cv_results_).to_csv(destination/"cv_results.csv",index=False)
                    pd.DataFrame({"row_position":rows[b],"true_label":y.iloc[b].to_numpy(),
                                  "predicted_label":pred}).to_csv(destination/"predictions.csv",index=False)
                    joblib.dump(fitted,destination/"model.joblib")
                    pd.DataFrame(records).to_csv(out/"fold_metrics.csv",index=False)
                    print(f"  F1={row['macro_f1']:.4f}; acute P={row['acute_precision']:.4f}, "
                          f"R={row['acute_recall']:.4f}; {elapsed:.1f}s",flush=True)
            assert (covered==1).all()
            pooled=[]
            for name,pred in oof.items():
                pd.DataFrame({"row_position":rows,"true_label":y,"predicted_label":pred}
                             ).to_csv(out/f"{name}_oof_predictions.csv",index=False)
                p,r,f,support=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
                cm=confusion_matrix(y,pred,labels=LABELS)
                write_json(out/f"{name}_pooled_metrics.json",{**metrics(y,pred),"rank_mae":rank_mae(y,pred),
                                                            "confusion_matrix":cm.tolist()})
                for i,label in enumerate(LABELS):
                    tp=int(cm[i,i]);fp=int(cm[:,i].sum()-tp);fn=int(cm[i,:].sum()-tp)
                    pooled.append({"model":name,"class":label,"support":int(support[i]),
                        "precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),
                        "tp":tp,"fp":fp,"fn":fn,"tn":int(cm.sum()-tp-fp-fn)})
            pd.DataFrame(pooled).to_csv(out/"pooled_per_class_metrics.csv",index=False)
            table=pd.DataFrame(records)
            measures=["macro_f1","accuracy","balanced_accuracy","acute_precision","acute_recall",
                      "rank_mae","search_seconds"]
            summary=table.groupby("model",sort=False)[measures].agg(["mean","std"])
            summary.columns=[f"{m}_{stat}" for m,stat in summary.columns]
            summary.to_csv(out/"summary.csv")
            winner=summary.macro_f1_mean.idxmax()
            lines=["# Feature-ablation exploratory comparison", "",
                "Same saved nested folds in seed 42's 7,200 training rows; original test rows excluded. "
                "Original SVM reference predictions reused exactly. Each new procedure has 12 inner-search "
                "candidates. Broader context/demographic inputs are separate from usage-only variants. No fresh test claim.","",
                "| Procedure | Macro-F1 (mean ± SD) | Acute precision (mean ± SD) | Acute recall (mean ± SD) |",
                "|---|---:|---:|---:|"]
            for name,row in summary.iterrows():
                cells=[f"{row[m+'_mean']:.4f} ± {row[m+'_std']:.4f}" for m in
                       ["macro_f1","acute_precision","acute_recall"]]
                lines.append("| "+" | ".join([name]+cells)+" |")
            lines += ["",f"Highest mean outer macro-F1: {winner}. This is a provisional exploratory ranking, "
                "not an independent significance result. Fold SD is not a confidence interval. "
                "Pooled class metrics cover 7,200 unique validation rows, including 84 acute records.","",
                "Models in fold directories were fitted only on that outer fold's training rows. "
                "No new model was refitted for deployment or evaluated on the original test set. "
                "Reference search time is zero here because its saved predictions were reused; "
                "it must not be interpreted as a zero-cost training method."]
            (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
            fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
            labels=["Six-feature SVM","+ context","+ demographic/platform","+ engineered","+ engineered/logs",
                    "Linear logistic","Interaction logistic"]
            for ax,m,title in zip(axes,["macro_f1","acute_precision","acute_recall"],
                                 ["Macro-F1","Acute precision","Acute recall"]):
                ax.bar(range(len(summary)),summary[m+"_mean"],yerr=summary[m+"_std"],capsize=3)
                ax.set_xticks(range(len(summary)),labels,rotation=55,ha="right",fontsize=8)
                ax.set_ylim(0,1.15);ax.set_title(title);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
            fig.suptitle("Feature ablations · exploratory nested validation · mean ± fold SD")
            fig.savefig(out/"feature_comparison.png",dpi=180);fig.savefig(out/"feature_comparison.svg")
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
    parser.add_argument("--output",type=type(ROOT),default=ROOT/"results/feature_ablation")
    parser.add_argument("--jobs",type=int,default=4)
    args=parser.parse_args();run(args.output,args.jobs)
