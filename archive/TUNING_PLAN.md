# Exploratory SVM and gradient-boosting study

Specified before running this study. Earlier baseline test scores have already
been viewed; this is exploratory and does not create a new untouched test set.

Use only the 7,200 training row positions saved for original seed 42. Exclude its
1,800 test positions from every fit, validation fold, and model selection below.
Some of these training records appeared in earlier holdouts for other seeds;
therefore the new validation results are not independent confirmation of an
entirely prespecified research program.

Use stratified three-fold outer CV (seed 2026), with stratified three-fold inner
CV (seed 2026 + outer fold index). Each inner search selects mean macro-F1, refits
on outer training rows, and predicts outer validation rows once. Preprocessing
and class-weight estimation stay inside fitting. Shared outer and inner folds
make the comparison paired. The three outer-fold SDs are not five-seed test SDs
and are not confidence intervals.

Compare three search procedures:

1. Original RBF-SVM search: the original 12 configurations, as a reference.
2. Expanded RBF-SVM: all original 12 plus 24 fixed randomly sampled additional
   configurations. Additional C in {0.1,0.3,1,3,10,30,100,300}; gamma in
   {scale,0.001,0.003,0.01,0.03,0.1,0.3,1}; class_weight in {None, balanced,
   acute-only weight 2, acute-only weight 4}. Other classes have weight 1 in the
   last two options. Sample without replacement, seed 2026.
3. Histogram gradient boosting: 36 configurations sampled without replacement,
   seed 2026; learning_rate in {0.03,0.06,0.1,0.2}, max_iter in {100,200,300},
   max_leaf_nodes in {7,15,31}, min_samples_leaf in {10,20,40},
   l2_regularization in {0,0.1,1,10}, and the same four class-weight choices.
   early_stopping=False so each configuration uses its specified iteration budget
   without a hidden internal validation split. random_state=2026 for reproducibility.

The two new procedures have equal candidate and CV-fit budgets: 36 candidates
times 3 inner folds per outer fold. They need not have equal wall-clock cost.
Freeze and save the exact sampled candidates before the first fit.

Primary comparison: outer-fold mean macro-F1 and pooled out-of-fold macro-F1.
Also report class-wise precision, recall, F1, TP/FP/FN/TN, macro precision,
balanced accuracy, accuracy, and fitting time. Preserve all candidate CV results,
selected parameters, folds, predictions, code/data hashes, and software versions.
Do not select thresholds from validation curves in this study. Acute weighting
may improve recall while reducing precision; report both, including other classes.

After comparison, select the new procedure (expanded SVM or boosting) with higher
mean outer macro-F1, breaking an exact tie in favor of expanded SVM. Run its fixed
candidate search on all 7,200 training rows using three-fold CV, refit and save a
model bundle. This is an exploratory model choice, not evidence that both precision
and recall improved. Do not evaluate this bundle on the original test rows or
replace the original baseline table.
