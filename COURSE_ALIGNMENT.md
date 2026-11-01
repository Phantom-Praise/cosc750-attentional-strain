# Course alignment audit

Reviewed October 26, 2026 against the four user-supplied lecture decks. These are
teaching references, not instructions to copy every example or practice exercise.
Page references below are physical PDF pages; lecture-slide numbers can differ
because some PDFs contain incremental overlays.

## What was already implemented

| Lecture concept | Connection to the implementation | Evidence / limits |
|---|---|---|
| Feature vectors, classes, supervised recognition | Six numeric usage measurements form x in R^6; labels are the four dataset-defined categories. | `run_baselines.py`: FEATURES, LABELS, load_data. This is classification, despite the name logistic regression. |
| Prior-only classifier | Majority reference chooses the most frequent training label. | DummyClassifier estimates its frequencies from training rows. It uses no feature evidence. |
| Linear discriminants and optimization | Multinomial logistic regression uses affine class scores followed by softmax and a class decision. | StandardScaler plus LogisticRegression with L2 regularization and the lbfgs solver. It minimizes a regularized conditional log-loss surrogate, not zero-one loss directly. |
| Margins, kernels, regularization | RBF-SVM learns nonlinear separating boundaries. C and gamma are selected by inner validation. | SVC uses one-versus-one classification internally. Its decision scores are not calibrated probabilities. A library solver handles the constrained optimization; no claim of a hand-coded KKT solver. |
| Bias-variance controls and ensembles | Logistic/SVM C, SVM gamma, tree depth and leaf size control flexibility; the forest combines 300 trees. | Validation selects these settings. Five-run score SD measures performance sensitivity, not the mathematical bias/variance of a predictor. |
| Honest evaluation | Five shared stratified outer holdouts and training-only stratified CV; scaling fitted inside each training fold. | Saved split positions, validation results, predictions, manifest, and integrity tests. Original test results are preserved. |
| More than accuracy | Macro-F1, balanced accuracy, per-class recall, and confusion matrices. | Prevents the common classes from fully hiding rare-class failures. |
| Feature relevance and dimensionality | Primary inputs are six usage variables rather than every available column. Cognitive measurements and identifiers are excluded. | This is a prespecified feature restriction, not learned feature selection or a completed ablation study. |

Relevant sources: introduction PDF pp. 13–17, 23–35, 37, 46–48, 51–58;
Bayes-optimal PDF pp. 19–27, 62–86, 131–171.

## What was added after reading the lectures

### Histogram Naive Bayes

`course_models.py` implements the lecture's sequence explicitly:

1. Estimate class priors as training class count / training sample count.
2. Fit shared equal-width feature bins on training data only. All classes use the
   same feature edges.
3. Count observations separately within each class and feature.
4. Estimate smoothed bin mass as `(count + alpha)/(class_count + alpha*K)`.
5. Divide by bin width to obtain a density height; the heights times widths sum to
   one within each class and feature.
6. Combine evidence under conditional feature independence, multiplying each
   feature density by the class prior. Computation uses log space for stability.
7. Normalize with log-sum-exp to obtain model posterior probabilities.
8. Minimize expected zero-one loss. This is equivalent to choosing the largest
   estimated posterior, subject to the deterministic tie rule.

The loss utility explicitly uses `loss[action, true_class]`; it is tested with a
known unequal-cost example. All reported experiment predictions use equal error
costs. No domain-specific cost matrix has been justified for the synthetic tiers.

Smoothing uses alpha in {0.5, 1}; alpha=1 is exactly add-one smoothing. This is the
multinomial/Dirichlet counterpart of the binary Beta smoothing described in the
estimation lecture. It does not implement full Bayesian integration over bin
boundaries, model choice, or every model parameter. Empirical class priors remain
frequency estimates. An unsmoothed alpha=0 path is tested against the classroom
example; it is not part of the main data search because empty bins can eliminate
all evidence for an observation.

The implementation searches a fixed finite set of bin resolutions. Histograms
can form a nonparametric density-estimation scheme when resolution grows with
sample size, but this finite experiment does not establish such an asymptotic
scheme or prove consistency (histogram lecture pp. 39–40).

`tests/test_course_models.py` reproduces the lecture's unadjusted posterior for
(70,78), its zero-bin example for (90,62), and the add-one posteriors of 0.6 Pass
for (90,62) and 4/7 Pass for (70,78). Tests also check density integrals,
normalization, boundary handling, and cost decisions.

Sources: histogram PDF pp. 10–16, 21–32, 36–42; estimation PDF pp. 7–15, 17–25;
Bayes-optimal PDF pp. 39–57; introduction PDF p. 31.

### Diagnostics beyond the original table

`run_course_extension.py` adds per-class precision, specificity, FPR, F1, explicit
empirical zero-one risk, one-versus-rest ROC-AUC, average precision, and ROC/PR
curves. A separate binary curve is needed for each class; the displayed curves
use acute versus rest for the prespecified seed 42. All five seeds' metrics and
curve coordinates are saved. Average precision is reported as AP, not relabeled
as trapezoidal PR-AUC. Per-seed Wilson intervals describe acute recall uncertainty
under a binomial approximation; overlapping test splits are not pooled to inflate
the sample size.

The original four models are refitted with their already-selected training-only
settings solely to recover scores. Their predicted labels are required to match
the original saved evidence exactly. No test curve is used to select a threshold.
For SVM, ROC and PR use ranking scores from decision_function; these are not
posterior probabilities. NB, forest, and logistic outputs are model probabilities
whose real calibration has not been established.

Sources: Bayes-optimal PDF pp. 161–194, including the imbalance/PR discussion in
the overlays of lecture slide 39; introduction PDF pp. 56–57.

## Findings and assumptions

The added histogram comparison achieves macro-F1 **0.6765 ± 0.0128** and acute
recall **0.7143 ± 0.1468** over the same five saved splits. The original SVM score
is **0.8387 ± 0.0207**. This is a useful course baseline even though its macro-F1
is lower. Higher acute recall alone does not establish that it is preferable:
its mean acute AP is 0.5400 versus 0.8627 for SVM and 0.8934 for logistic regression.

Training-only, within-class correlation diagnostics show substantial dependence.
For example, in seed 42's severe-class training rows, daily short-form hours and
bedtime scrolling have Pearson r about 0.945. That contradicts an exact conditional
independence assumption for those features in this sample and gives a concrete
reason to question NB's factorization. It may contribute to the lower macro-F1;
the correlation alone does not prove why the score differs or measure probability
miscalibration. Bin resolution, smoothing, and boundary clipping also matter.

The new method was added because the supplied lectures emphasize it, after the
original test scores had been seen. This is documented in COURSE_EXTENSION_PLAN.md.
It is an exploratory extension; do not present it as an independently confirmed
method selected before all test observations.

## Topics not added, and why

| Topic | Reason / boundary |
|---|---|
| Exact true Bayes-optimal predictor and Bayes risk | Cannot compute them without the true joint distribution or an adequate verified data generator. Estimated NB posteriors are not that distribution. Its errors are not evidence of reaching the Bayes floor. |
| Formal bias-squared + variance + noise calculation | The familiar exact decomposition in the slides is for squared-loss regression with a known expectation structure. Five macro-F1 scores do not estimate its components. Use theory to discuss flexibility and report sensitivity honestly. |
| Binary Beta-Binomial flower model | Our six features are continuous and the label has four classes. Arbitrary binary cutoffs would discard information. Histogram count smoothing provides the relevant multicategory analogue. |
| Full joint six-dimensional histogram | Even 10 bins per feature creates 10^6 joint cells per class, with only 84 acute training examples per outer split. Separate histograms avoid that sparsity by assuming conditional independence. |
| Equal class priors | The classes are highly imbalanced. Use training-frequency priors, not the equal priors chosen for balanced teaching examples. They estimate synthetic benchmark proportions, not population prevalence. |
| Clinical asymmetric costs or a reject action | Costs and rejection thresholds need a defensible application definition. The decision utility supports a square loss matrix, but experiments retain zero-one loss; no reject action was implemented. |
| PCA, LDA projection, explicit eigendecomposition/SVD | Feasible, but six features are already interpretable. The training correlation audit flags redundancy without yet changing the feature representation. A projection would be a separately validated experiment; no claim it cannot help. |
| Gaussian density classifiers / Gaussian NB | Feasible alternatives; not required to illustrate the supplied histogram lesson. Histogram NB adds a directly traceable generative comparison without adding every possible learner. |
| Deep networks and double descent | These slides explain theoretical regimes, not a requirement to train a network. No experiment yet establishes that additional architecture/search complexity helps this small tabular study. |
| Clustering, EM, reinforcement learning | Different objectives from predicting four provided class labels. They could support future analysis but are not needed for the current supervised comparison. |
| Segmentation, image/signal processing, invariance engineering | The CSV already supplies aggregate measurements. We do not have raw video, images, or time series to segment or transform. |
| Group-aware or chronological splitting | No repeated participant IDs, timestamps, or known group keys are present. Row-level stratification is a benchmark assumption; unique IDs alone do not prove independence or absence of hidden groups. Do not invent a timeline from the ID. |
| Teacher's illustrative 60/20/20 split and 5/10 folds | Our outer 80/20 split with inner three-fold CV separates the same three roles. Each training CV round fits on 4,800 and validates on 2,400 records. Three folds preserve the originally fixed compute budget and give 28 acute validation examples per fold. The slides call 5/10 folds typical, not mandatory. |
| Calibration, feature-group ablations, future deployment checks | Not completed by this review. They need separately specified analysis; normalized posteriors, correlation tables, or random holdouts cannot substitute for them. |
| Hand-coded gradient descent, Hessians, or KKT optimization | Existing library solvers perform the needed optimization; the mathematical topics explain their objectives. They are not separate data-analysis features to implement merely because the slides derive them. |

## Remaining submission work

Document the method, objectives, decision rules, assumptions, algorithm, and
complexity in the new paper section. Histogram NB is a classical course baseline;
its addition does not by itself settle the separate strong published domain-method
requirement, establish novelty, or complete the proposed-method choice. Update
the AI-use disclosure to cover the custom histogram code and course audit.

Sources:

- [Introduction and mathematical background](/Users/praiseben/Downloads/lec01_introduction_math_background.pdf)
- [Parameter estimation for classification](/Users/praiseben/Downloads/classification_estimation.pdf)
- [Bayes-optimal predictor and honest evaluation](/Users/praiseben/Downloads/lec02_bayes_optimal.pdf)
- [Histograms and Naive Bayes](/Users/praiseben/Downloads/histograms_naive_bayes.pdf)
