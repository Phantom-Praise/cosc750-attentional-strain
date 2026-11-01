# Related research and code review

Reviewed October 26, 2026. Scope: inspect public implementations, compare their
assumptions with our six-feature, four-class synthetic task, and identify justified
next experiments. No external repository was executed, installed, or incorporated
into the training code during this review. No new model performance was measured.

## Main recommendation

Try an ordered-label model and a small preprocessing comparison before another
large unconstrained model search. These target identifiable structure in our data.
Threshold tuning is a separate option for choosing an acute precision/recall trade-off.
None of these suggestions guarantees an improvement.

## 1. Ordinal regression: the most task-specific new hypothesis

Reviewed [fabianp/mord](https://github.com/fabianp/mord), especially
[`mord/threshold_based.py`](https://github.com/fabianp/mord/blob/c55c599f009f3855fb0fb89467a7223330db4ab7/mord/threshold_based.py).
The project accompanies research by Pedregosa, Bach, and Gramfort:
[On the Consistency of Ordinal Regression Methods](https://arxiv.org/abs/1408.2327),
published in JMLR in 2017. The implementation includes all-threshold and
immediate-threshold logistic models, a regularized linear score, ordered cutpoints,
and class probabilities computed from cumulative probabilities.

Why relevant: normal, mild, severe, and acute have a natural severity order.
Our nominal models do not explicitly encode it. In the already-saved nested
validation predictions for the original SVM, all 900 errors among 7,200 unique
rows are one category apart. There are 6,300 correct predictions and no errors
two or three categories apart. This is evidence motivating an ordinal hypothesis,
not proof of a one-dimensional true generator or guaranteed improvement.

Proposed adaptation: explicitly map normal=0, mild=1, severe=2, acute=3. Fit a
regularized ordinal classifier with scaling learned within each training fold.
Tune regularization and any class weights by inner macro-F1; compare against the
same original SVM procedure on the same outer folds. Keep macro-F1 and all
class-wise precision/recall metrics; rank MAE can be an additional diagnostic.
The ordinal order does not imply equal clinical distances between categories.

Implementation caveats: this is older code. Its estimator classes do not provide
the modern classifier mixin used in our wrappers, the labels must be consecutive
integers, and optimization failures need explicit checking. Adapt and test the
idea rather than dropping a legacy estimator unexamined into our current pipeline.
The all-threshold model's default score is negative absolute error, so it must
not silently replace our macro-F1 selection criterion.

## 2. Tabular benchmark: vary representation, not only C and gamma

Reviewed Grinsztajn, Oyallon, and Varoquaux's
[paper](https://arxiv.org/abs/2207.08815) and
[LeoGrin/tabular-benchmark](https://github.com/LeoGrin/tabular-benchmark).
The inspected code includes
[`gaussienize`](https://github.com/LeoGrin/tabular-benchmark/blob/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d/src/data_transforms.py#L97),
which offers standard, robust, normal-quantile, uniform-quantile, and power
transformations. It fits transformations on training rows and applies them to
validation and test rows. The benchmark's datasets and metrics differ from ours;
its published scores do not predict performance on our CSV.

Proposed adaptation: compare StandardScaler, RobustScaler, and a normal-output
QuantileTransformer in the SVM pipeline, retaining the six inputs. Quantile
transformation changes feature spacing and therefore RBF distances; it can also
compress meaningful tail differences, so inspect acute performance. Select the
transformer inside inner CV, never before splitting. Do not add outcome-related
cognitive variables just to obtain higher scores. Copying old API calls wholesale
is inappropriate; the repository's documented environment is Python 3.8.

### A narrower future boosting experiment

The benchmark's
[`hgbt_config.py`](https://github.com/LeoGrin/tabular-benchmark/blob/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d/src/configs/model_configs/hgbt_config.py)
uses a 1,000-iteration ceiling, early stopping, a 20% training-internal validation
fraction, patience of 20, and shallow-depth candidates. Our completed study fixed
100–300 iterations and disabled early stopping. Thus our result only establishes
that the tested boosting search did not improve macro-F1, not that boosting cannot
help. A separate low-learning-rate, shallow, early-stopped experiment is plausible;
its stopping validation must be inside the current training fold. With few acute
examples, stopping based on log loss can disagree with macro-F1, so record both.

## 3. Imbalanced-learn: controlled resampling rather than weights alone

Reviewed the official
[`plot_pipeline_classification.py`](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/examples/pipeline/plot_pipeline_classification.py),
the [ensemble comparison](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/examples/ensemble/plot_comparison_ensemble_classifier.py),
and SMOTE's implementation. The examples place sampling inside fitting and compare
ordinary ensembles with balanced variants. The SMOTE implementation handles
multiclass data one class at a time. Their example datasets are not ours.

Potential experiment: modest minority oversampling or balanced forests, retaining
the original class proportions in all validation folds. Use an imblearn pipeline
with scaling before neighbor-based sampling. For our four classes, specify desired
class counts via a dictionary or fold-aware callable, not a binary-only float
sampling ratio. Choose those counts from the training fold only. SMOTE creates
fractional synthetic feature values, including for session-count inputs; evaluate
whether those interpolations are defensible. Random oversampling is a useful
simpler reference. Fully balancing to the majority count is not automatically
appropriate given only 84 acute records in the outer training partition.

Priority is lower than ordinal modeling/preprocessing: our heavier class weights
already demonstrated more recall can cost substantial precision. These examples
are not evidence that resampling will solve that trade-off. The official
[pitfall guide](https://imbalanced-learn.org/stable/common_pitfalls.html) explicitly
explains why resampling the whole dataset before splitting leaks information.

## 4. Scikit-learn: change the action rule separately from the model

Inspected the actual
[`TunedThresholdClassifierCV` implementation](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/model_selection/_classification_threshold.py#L709).
It fits across CV splits, evaluates candidate cutoffs, and selects the threshold
using validation scores. Its prediction path is binary; we cannot directly wrap
our four-class model and assume it solves multiclass decision selection.

Potential adaptation: a separately specified acute-versus-rest decision followed
by a clear rule for assigning normal/mild/severe when acute is not selected.
Choose a prespecified objective, such as acute F1 or recall subject to an agreed
validation precision floor. Select cutoffs from inner out-of-fold predictions,
then measure the complete four-class rule on outer validation rows. Do not select
a cutoff from the existing test ROC/PR curves. Changing a cutoff alone typically
changes the precision/recall balance rather than creating new class information.

If probability-based expected-risk decisions are desired, first assess training-
only calibration. The official
[calibration guide](https://scikit-learn.org/1.7/modules/calibration.html) notes
small-sample overfitting concerns with isotonic calibration. Sigmoid calibration
is a sensible initial comparison here, but neither calibration nor a normalized
probability guarantees higher macro-F1. Current SVM decision scores are not
probabilities. Calibration must also remain inside the validation hierarchy.

## 5. Closely related social-media study: relevant topic, different input

Reviewed the method and appendices in Peters et al.'s
[Social Media Use is Predictable from App Sequences](https://arxiv.org/html/2404.16066v1).
The paper models ordered app events using LSTMs and Transformers, compares pooled
and personalized approaches, and includes weighted-loss experiments. It identifies
an [OSF code page](https://osf.io/rkswe/), but the public web reader could not access
it and a direct OSF API read returned HTTP 401. Its code was therefore not inspected.

Our CSV contains aggregate usage measurements without the ordered app-event logs
needed to reproduce its representation. Its binary social-media-use objective
also differs from our four synthetic strain tiers. Its scores are not comparable
to ours. It supports future collection of sequential data, not constructing fake
sequences from participant IDs or claiming a direct LSTM reproduction now.

## Exact-dataset search limits

Queries included the exact CSV filename, its basename, cognitive_decay_tier, and
the Kaggle dataset slug in web/GitHub searches. No verifiable exact-dataset
predictive codebase was identified in this limited search. This does not prove
none exists, especially among unindexed Kaggle notebooks or private projects.
No dataset-specific state-of-the-art score or strongest published baseline was
established. Do not import claimed high scores from unrelated datasets.

## Recommended next comparison

1. Original SVM reference, unchanged.
2. An ordinal logistic approach with explicit label order and modern estimator
   integration, with training-only regularization selection.
3. SVM with training-only standard/robust/quantile preprocessing selection.

Use the existing training-only nested validation framework and keep the test
table intact. Freeze candidate lists and budgets before fitting. Report this as
an exploratory extension, since earlier test and validation results influenced
the research direction. Add threshold/calibration or resampling as later focused
ablations if the first comparisons justify them, not a large untracked search.

For the assignment, the ordinal method provides a traceable published method and
author code. That provenance does not establish it as the strongest method on our
dataset or itself make our adaptation novel. Any reused code must retain its
license/attribution and enter the AI/reuse disclosure appropriately.

## Evidence available locally

- `../results/tuning/original_svm_oof_predictions.csv`: 7,200 unique validation
  records used for the adjacent-error calculation above.
- `review_manifest.json`: inspected files, revisions, access limits, and the hash
  of the local prediction evidence.
- Existing experiment code/results were not changed by this research review.
