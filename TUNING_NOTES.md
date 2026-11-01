# Tuning implementation notes

This experiment compares search procedures through nested validation. An original
12-setting SVM search is the reference; expanded SVM and histogram gradient
boosting each get 36 settings per inner search. Larger search spaces are not
guaranteed to yield better outer validation scores: an inner-CV winner can fit
validation noise, which is why the outer layer is needed.

The first run stopped when the installed gradient-boosting implementation applied
a dictionary keyed by original string labels to internally encoded numeric
labels. No boosting outer-fold result was produced in that attempt. Its partial
evidence is retained in `results/failed_tuning_class_weights/` and excluded from
the final comparison.

`tuning_models.py` addresses this by computing sample weights from the original
labels inside each estimator fit, and passing those weights to the booster with
its own class_weight disabled. This preserves the planned weights. In particular,
balanced weights are calculated from the current training fold, rather than once
on the full dataset. A test checks both custom and balanced weights with string
labels. The complete study restarted from the first outer fold with the same
candidate sampling seed and search protocol.

A subsequent partial run was stopped before any boosting outer-fold score was
produced because the threaded search oversubscribed OpenMP workers. It is retained
in `results/interrupted_tuning_threads/`. The runner now fixes OMP_NUM_THREADS and
OPENBLAS_NUM_THREADS before importing numerical libraries, in addition to runtime
threadpool limits. The study restarted with the same candidates and folds; this
changes resource control, not the statistical protocol.

The model bundle uses an importable class from `tuning_models.py`, so it can be
loaded in another Python process. Use the pinned environment and load only trusted
joblib files. The bundle includes the six feature names, label names, training row
positions, and the fact that the model choice is exploratory.

Interpretation boundaries:

- Nested validation scores are not directly comparable to earlier five-split test
  averages, which use different evaluation rows and training sizes.
- The original seed 42 test rows are excluded throughout this study, but the
  project has already seen earlier holdout results. This is not a pristine final
  evaluation of a research program designed without any test feedback.
- Macro-F1 is the primary objective. Acute precision and recall may move in
  opposite directions, even if macro-F1 increases.
- Class weights change the training loss. They do not establish clinical error
  costs or change the reporting definition of precision and recall.
- Selecting the model family by its outer-fold mean makes that selection
  exploratory too. A saved model is not a new independent performance estimate.
- The original baseline code, protocol, and saved metrics are preserved.

AI-use disclosure should include this additional search orchestration and the
fold-local sample-weight wrapper.

## Observed outcome

The original SVM search retained the highest mean outer macro-F1: 0.8462, compared
with 0.8391 for expanded SVM and 0.8311 for gradient boosting. This study provides
no demonstrated overall improvement from the larger search or added model.

Across pooled out-of-fold acute predictions, original SVM had TP=59, FP=11, FN=25;
expanded SVM had TP=59, FP=18, FN=25; boosting had TP=66, FP=33, FN=18. Boosting
caught seven more acute records but produced 22 more acute false alarms. The
mean-fold acute recall rose from 70.24% to 78.57%, while mean-fold acute precision
fell from 84.29% to 67.46%. Pooled precision and mean-fold precision need not match.

Recommendation: retain the original SVM as the leading comparison; do not claim
the new models improved both precision and recall. These exploratory comparisons
are useful evidence of the bias/flexibility and class-weight trade-offs.

The saved `results/tuning/model.joblib` is explicitly an experimental refit of the
better of the two new search procedures, as prespecified, not a replacement for
the leading original SVM. That procedure was expanded SVM; its full-training CV
selected C=1, gamma=scale, and no class weights. This refit has no new test score.
Its selection can differ from the earlier original search because the training
CV partitions differ; it does not establish those parameters as universally best.
