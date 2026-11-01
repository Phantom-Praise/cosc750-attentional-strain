# Stage 3 experiment protocol

Status: specified before model fitting or test-set evaluation. No results have been generated.

## Data and task

- Source file: `short_form_brain_rot_cognitive_decay.csv`
- SHA256: `2f0c2870adb1ccf7c3e3b9a35eb68e71c912868fd91068d3f81be7ba3a0b49c2`
- Observed shape: 9,000 records, 19 columns; no missing cells or duplicate complete rows.
- Target: `cognitive_decay_tier`, four classes.
- Primary features: `daily_short_form_hours`, `avg_clip_duration_sec`, `swipes_per_minute`, `scroll_sessions_day`, `bedtime_scroll_min`, `total_screen_time_hours`.
- Exclude the identifier, target, demographics, sleep, multitasking, and cognitive measurements from the primary predictors.
- Records represent synthetic observations; results do not establish clinical validity or real-world generalization.
- Learn a multiclass decision rule under zero-one classification loss. Macro-F1 is the model-selection criterion, chosen to give all four classes equal weight; it is not equivalent to minimizing zero-one loss.

## Comparisons

1. Majority-class predictor: trivial reference; no tuning.
2. Multinomial logistic regression: linear classical baseline.
3. RBF support vector classifier: nonlinear margin-based comparator.
4. Random forest: nonlinear tree-ensemble comparator.

The primary study is a controlled comparison, not yet a claim of a novel algorithm. Before drafting the method section, specify which method or justified modification is the proposed approach. These model families also do not, by themselves, establish compliance with the requirement for a strong published method: verify the relevant publication and its applicability, and implement its applicable procedure. Do not describe a model as the strongest published method on this dataset without evidence.

## Partitions and selection

- Use five prespecified seeds: 42, 43, 44, 45, 46.
- For each seed, make an 80% training / 20% test stratified split.
- All models share exactly the same partition for a given seed.
- Use shuffled, stratified three-fold cross-validation on the training portion only.
- Select hyperparameters by mean validation macro-F1, then refit on the complete training portion and evaluate once on its held-out test portion.
- Each learned model receives 12 parameter configurations, or 36 validation fits per seed.
- Fit scaling inside the validation pipeline for logistic regression and SVM; forest needs no scaling. No resampling in the primary comparison.
- Class weights, when used, are computed from the current training fold only.
- Do not revise features, search spaces, or methods using test results. Repeated holdouts have overlapping test observations and are not independent external replications.

## Search spaces

- Logistic regression: C in {0.01, 0.1, 1, 10, 100, 1000} crossed with class_weight in {None, balanced}; L2 regularization, lbfgs solver, max_iter=5000. Record and resolve convergence warnings before reporting results.
- RBF SVM: C in {0.1, 1, 10}, gamma in {scale, 0.1}, class_weight in {None, balanced}; probability=False. Decision is the implementation's multiclass voting rule, not posterior risk minimization.
- Random forest: n_estimators=300; max_depth in {None, 10, 20}, min_samples_leaf in {1, 5}, class_weight in {None, balanced}.
- Resolve equal validation scores by a fixed documented configuration order.

## Reporting

- Primary: macro-F1 with all four labels explicitly included and zero_division=0.
- Secondary: balanced accuracy, accuracy, recall for each class, and confusion matrices in a fixed label order.
- Report every per-seed score and mean plus sample standard deviation over five runs.
- This standard deviation describes sensitivity to splits, selection, and model randomness; it is not a confidence interval.
- Record search time, refit time, prediction time, selected parameters, hardware, library versions, and partition identifiers.
- No significance claim is required for this preliminary comparison; overlapping repeated holdouts would require care in any later inferential test.
- Record failures and warnings; do not silently discard unsuccessful seeds.

## Later stages of this submission

- Verify dataset provenance and available label-generation documentation before drawing leakage conclusions.
- Keep any expanded cognitive-feature experiment separate from the primary comparison; a performance gain alone cannot prove leakage.
- Supply a repository README, pinned environment, and one command reproducing the baseline table.
- Extend the cumulative paper by 2–3 pages and extend the existing slides. Check whether the 6–7-page target excludes references.
- Prepare an 8–10-minute recording focused on this stage and identify any changes to earlier stages.
- Update the feedback response and AI-use disclosure based on actual work and feedback received.
- ZIP the final paper, slides, and actual recording; include repository access information. A script does not substitute for the requested recording.
