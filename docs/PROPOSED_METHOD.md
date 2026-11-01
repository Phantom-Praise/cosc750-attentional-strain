# Proposed approach: calibrated RBF-SVM with validation-selected acute decision rule

Status: selected after exploratory comparisons on October 26, 2026. This
specification describes the implemented procedure, not a new unrun model.

## 1. Problem and representation

Let D = {(x_r,y_r)} contain n labeled records with x in R^6. The six entries are,
in order: daily short-form hours, average clip duration in seconds, swipes per
minute, daily scrolling sessions, bedtime scrolling minutes, and total screen
hours. There are four dataset-defined classes: normal, mild, severe, and acute.
Their recorded names and machine-readable feature names remain those in the CSV.

The primary method excludes participant ID, demographics/platform, sleep,
multitasking, and the six cognitive measurements. Context and demographic
experiments remain separate ablations. Inputs are synthetic aggregate usage
measurements plausibly obtainable from logs; they are not actual collected logs.
Do not interpret predictions as diagnoses or causal effects of scrolling.

Zero-one loss is L(a,y)=1[a != y]. The conditional risk under a posterior is

\[
R(a\mid x)=\sum_{j=1}^{4}L(a,\omega_j)P(\omega_j\mid x).
\]

With the true posterior and equal error costs, minimizing this risk gives the
largest posterior. We do not know that true posterior. Macro-F1 is the primary
selection metric because it averages class F1 equally; it is not identical to
minimizing zero-one risk. Accuracy estimates one minus empirical zero-one loss.

Assume row-level observations are exchangeable for this synthetic benchmark.
Unique IDs do not prove independence or rule out hidden groups. Random splits do
not assess future temporal or population shift. Class priors and label generation
are dataset-specific, not estimates of real-world clinical prevalence.

## 2. Base learner and objective

Each fitted calibration member learns its own training-only feature mean mu and
standard deviation sigma, with z_j=(x_j-mu_j)/sigma_j. StandardScaler handles a
constant feature with scale one. No preprocessing is fitted on validation rows.

The RBF kernel is

\[
K(z,z')=\exp(-\gamma\lVert z-z'\rVert_2^2).
\]

SVC fits one-versus-one binary separators. For each class pair, its soft-margin
objective has the form

\[
\min_{w,b,\xi}\frac12\lVert w\rVert^2+C\sum_r\xi_r,
\quad t_r(w^T\phi(z_r)+b)\ge1-\xi_r,\quad\xi_r\ge0.
\]

C controls the error penalty and gamma controls kernel locality. The proposed
implementation uses no class weights or resampling. SVC's multiclass decision
scores used for calibration are not probabilities. The pairwise training and
multiclass score construction are provided by scikit-learn/libsvm.

## 3. Sigmoid calibration ensemble

Inside each training fit, split the available rows into R=3 stratified calibration
folds (shuffle=True, seed 2026). For member m, fit the scaling/SVM pipeline on
the other two folds, then fit per-class sigmoid calibrators on held-out scores.
Conceptually, a calibrator has the form

\[
q_{mi}(x)=\{1+\exp[A_{mi}f_{mi}(x)+B_{mi}]\}^{-1}.
\]

The library estimates sigmoid parameters from held-out binary class targets
using its calibration implementation. Normalize the class outputs within each
member and average its probabilities across the three members:

\[
p_{mi}(x)=\frac{q_{mi}(x)}{\sum_jq_{mj}(x)},\qquad
\widehat p_i(x)=\frac13\sum_{m=1}^{3}p_{mi}(x).
\]

The stored estimator therefore contains three fitted pipeline/calibrator pairs;
it is not a single SVM refitted on all rows after calibrating. These are model
probability estimates. We have not established their real-world calibration,
nor completed a comparative Brier-score/reliability analysis.

Implementation contract: CalibratedClassifierCV(method="sigmoid", ensemble=True,
cv=StratifiedKFold(3, shuffle=True, random_state=2026), n_jobs=1), with
Pipeline(StandardScaler(), SVC(C=C, gamma=gamma)). Scaling is inside the
calibration estimator, not outside the calibration folds.

## 4. Explicit four-class decision rule

Let A denote the acute class. The candidate decision rules are:

\[
\widehat y(x)=
\begin{cases}
\arg\max_i\widehat p_i(x), & \tau=\mathrm{None},\\
A, & \tau\ne\mathrm{None}\text{ and }\widehat p_A(x)\ge\tau,\\
\arg\max_{i\ne A}\widehat p_i(x), & \text{otherwise}.
\end{cases}
\]

The None rule is minimum estimated zero-one risk. A finite cutoff is an
empirical macro-F1-selected acute operating rule; do not call it minimum
zero-one risk or claim it implements validated clinical costs. Ties follow the
estimator's fixed class order. Find the acute column by its label, not by assuming
that its probability column is last in the library's sorted class array.

## 5. Selection and algorithm

Use C in {1,10,100}, gamma in {scale,0.03}, and tau in {None,0.2,0.35,0.5}: 24
configurations. The scale choice is computed inside each fitted standardized SVM,
not globally from the dataset. Select the highest mean inner three-fold macro-F1;
ties use the deterministic saved candidate order.

```text
Algorithm: training and outer validation
Input: six-feature records and the saved outer/inner stratified folds.
For each outer fold:
  For each of the 24 candidate (C, gamma, tau) configurations:
    For each inner fold:
      Within its training rows, fit three scaling/SVM/calibrator pairs.
      Predict its inner validation rows using the candidate decision rule.
    Average the inner macro-F1 scores.
  Select the configuration with highest mean inner macro-F1.
  Fit its three-member calibrated estimator on the outer training rows.
  Predict the outer validation rows once; save predictions and metrics.
Output: three independently outer-fitted estimators and out-of-fold predictions.
```

The three outer folds and their three inner folds are the same saved partitions
as the original SVM comparison, restricted to seed 42's 7,200 training rows. Each
outer validation set has 2,400 records, including 28 acute records. The original
1,800 test rows are excluded throughout this proposed-method study.

There is no final new test score or automatically deployed full-training model.
Later fitting a single submission model would require selecting settings within
the available training rows; it must not select C/tau from outer test performance.
The overall decision to study and select this family is exploratory because
earlier test and validation results were already observed.

## 6. Course-theory justification and complexity

The RBF kernel allows nonlinear boundaries while the margin penalty controls
flexibility. Calibration makes an estimated-posterior decision rule available,
connecting the method to Bayesian decision theory. The acute rule makes the
precision/recall operating choice explicit. Calibration does not add measurements
or guarantee better accuracy, calibrated probabilities, or a lower Bayes error.

Let T_SVM(n,d) be base SVM fitting cost and S the number of support vectors in
one member, treating the four classes as fixed. Base prediction is O(Sd) per
record up to fixed multiclass factors. Three members multiply prediction and
stored-model cost by about R=3; sigmoid/normalization adds O(Rc) per record.
Stored support-vector data require O(RSd), apart from solver/cache overhead.

One fixed calibrated fit costs roughly R*T_SVM(2n/3,d), plus scoring and
calibration. With H candidates and K inner folds, selection has leading cost
O(HKR*T_SVM(2(K-1)n/(3K),d)), plus scoring, calibration, and the final calibrated
refit. Kernel SVM fitting can be superquadratic in sample count, with worst-case
cubic behavior for conventional dense optimization; actual libsvm scaling depends
on support vectors, tolerance, cache, and data. Do not equate these worst-case
bounds with measured wall-clock time or assume a full kernel matrix is stored.

Relative to B4's one fitted SVM, P adds calibration ensemble and search cost. The
reported proposed search has H=24 versus B4's original H=12, and internally fits
more SVMs. It is not an equal-compute comparison or a component-isolating ablation.

## 7. Preliminary finding and permitted claims

Mean outer macro-F1 is 0.8492 (SD 0.0134) for P versus 0.8462 (SD 0.0152) for B4.
Mean acute precision is 0.8551 versus 0.8429; mean acute recall is 0.7024 for both.
The selected outer-fold settings were (C=100,gamma=0.03,tau=None),
(10,0.03,None), and (10,0.03,0.2). These are validation-selected fold-specific
settings, not three competing choices to select using their outer scores.

Describe this as a small preliminary difference, not statistically established
superiority or improvement in both precision and recall. No numeric ceiling,
clinical validity, causal interpretation, or algorithmic novelty is claimed.
The contribution is the restricted-input, reproducible comparison and explicit
exploratory calibration/decision-rule study on this synthetic benchmark.

## Implementation and sources

- Code: `../research_models.py`, class CalibratedAcuteSVM.
- Search, folds, and results: `../run_research_experiments.py` and
  `../results/research_experiments/`.
- Frozen search plan: `../RESEARCH_EXPERIMENT_PLAN.md`.
- [Cortes and Vapnik (1995)](https://doi.org/10.1007/BF00994018).
- [Calibration overview](https://scikit-learn.org/1.7/modules/calibration.html).
- [Exact ensemble behavior](https://scikit-learn.org/1.7/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
- [SVC implementation documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html).

The solver, calibration, and preprocessing are reused library implementations.
The orchestration, wrappers, and drafting were generated with Codex and require
the student's review and AI/reuse disclosure.
