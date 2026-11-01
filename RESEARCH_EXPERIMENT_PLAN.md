# Research-inspired exploratory experiment plan

Frozen before fitting these new methods. Existing test and validation results
informed this direction; this is exploratory, not fresh confirmatory testing.
No original test rows will be evaluated or used for fitting or selection.

Reuse exactly the three outer folds and three inner folds recorded by
`results/tuning/fold_*/splits.json`, within seed 42's 7,200 training rows.
Reuse the original SVM's saved outer predictions as the paired reference.
Each new procedure receives 24 configurations, selected by inner macro-F1.
Save exact configuration lists before fitting. Candidate budgets are equal;
base-estimator fitting cost is not equal, particularly for calibration ensembles.

## New procedures

1. Ordinal logistic: use mord 0.7's published immediate-threshold objective and
   gradient, with a modern sklearn wrapper and explicit optimization checks.
   Normal=0, mild=1, severe=2, acute=3. Alpha in
   {0.001,0.01,0.1,1,10,100,1000,10000}; class weights {None, acute2, acute4}.
   Scale within fitting. Use ordered threshold predictions, matching mord's
   immediate-threshold classifier, and expose cumulative-model probabilities.
2. SVM preprocessing: {standard, robust, normal-quantile} crossed with
   C={0.1,1,10,100} and gamma={scale,0.03}; no class weights. Fit each transformer
   inside the current fit. Quantile transformer uses at most 100 quantiles and
   random_state=2026. No cognitive features or new demographic inputs.
3. Oversampled SVM: {random oversampling, SMOTE} crossed with acute-class training
   count multipliers {2,4,8} and C={0.1,1,10,100}, gamma=scale. Other classes stay
   unchanged. Scale first; sample only during fitting. SMOTE k_neighbors=3.
   Counts come from the current training fold. Fractional synthesized session
   values are an explicit approximation, not real new participants.
4. Calibrated SVM acute decision rule: C={1,10,100}, gamma={scale,0.03},
   cutoff={None,0.2,0.35,0.5}. Fit a pipeline of scaling and SVM inside sigmoid
   CalibratedClassifierCV with three stratified folds, nested inside each current
   training fit. None uses four-class probability argmax. Otherwise predict acute
   when calibrated acute probability >= cutoff, else argmax over the other three.
   No claim that each cutoff corresponds to a clinical error-cost threshold.
5. Early-stopped boosting: max_iter=1000, early_stopping=True, validation_fraction=.2,
   n_iter_no_change=20, scoring=loss, random_state=2026. Sample 24 configurations
   without replacement (seed 2026) from learning_rate={.01,.03,.06,.1},
   max_depth={2,3,4}, max_leaf_nodes={7,15,31}, min_samples_leaf={10,20,40},
   l2_regularization={.1,1,10}, weights={None,acute2,acute4}.
   Apply sample weights within each fit before internal early-stopping splitting.
6. Balanced random forest: sample 24 configurations without replacement, seed 2026,
   from n_estimators={100,300}, max_depth={None,10,20}, min_samples_leaf={1,5,10},
   max_features={sqrt,0.8}. sampling_strategy=all, replacement=True, bootstrap=False.
   This deliberately tests the balanced-ensemble suggestion separately from SMOTE.

## Reporting and limits

Report outer-fold mean and sample SD of macro-F1, accuracy, balanced accuracy,
acute precision/recall, and runtime; also all class-wise pooled OOF metrics and
TP/FP/FN/TN for 7,200 unique validation rows. Include ordinal rank MAE as a
secondary diagnostic, not a replacement for macro-F1. Save CV candidates,
selected parameters, row memberships, optimization/stopping details, predictions,
versions, hashes, and timing. Keep original experiment outputs unchanged.

No family or threshold is selected from outer scores before prediction. After
comparison, name the highest mean-macro-F1 procedure provisionally, including the
original SVM reference. Do not infer significance from fold SD or claim improved
precision and recall unless both actually rise. Do not test or deploy a new model
automatically. A new independent evaluation would still be needed for a final
claim following this exploratory model comparison.

The sequence-based social-media model is not implemented because the available
CSV has no ordered app-event logs. No fake sequences will be constructed.
References and method provenance are in `research/RELATED_CODE_REVIEW.md`.
