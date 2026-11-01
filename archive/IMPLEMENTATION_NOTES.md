# Implementation and audit notes

## Run context

- Coding session: October 26, 2026.
- Hardware observed with macOS sysctl: Apple M3 Pro, 11 CPU cores, 18 GiB memory.
- Runtime: Python 3.12 virtual environment in `.venv`; all resolved packages pinned
  in `requirements.txt`. The result manifest records the exact versions and platform.
- The local project sits below an existing home-directory Git repository. No remote
  repository was created or published during this baseline experiment.

## First-run correction

The first invocation stopped during logistic-regression cross-validation because
the scoring wrapper validated the default numeric positive label against string
class names. The macro-F1 scorer now explicitly sets `pos_label=None`; the metric
still uses the same fixed four labels, macro averaging, and zero-division rule.
An integration test checks scoring with these string labels. The incomplete first
run is retained at `results/failed_initial_scorer/` with a failed manifest; it is
excluded from the baseline summary. The full comparison restarted from seed 42.
No predictor, split, metric, or hyperparameter search was changed in this correction.

## Source checks and open questions

- The Kaggle dataset URL resolves, but its documentation text was not available
  through the web reader. The local data checksum and shape are independently
  checked; synthetic generation, licensing, and label-construction details still
  require a direct documentation audit.
- Publisher records for Cortes and Vapnik (1995), DOI 10.1007/BF00994018, and
  Breiman (2001), DOI 10.1023/A:1010933404324, verify the original method citations.
  The code uses scikit-learn implementations with the documented protocol rather
  than claiming an exact reproduction of either original paper's experiment.
- No dataset-specific strongest published baseline was established by the limited
  search. A strong published-method requirement remains open for the submission.
- The baseline run does not designate its test-set winner as a prespecified novel
  proposed method. A method decision based on these results is exploratory and
  must be described as such; further tuning cannot reuse these tests as unseen data.

## AI disclosure to carry into the paper

Codex generated the experiment orchestration, preprocessing/model setup, data
checks, reporting figures, README, and validation tests with user-approved choices
of predictors and comparison protocol. The student should review and understand
the code, verify citations and dataset provenance, and reproduce the results.
Claims of independent student authorship of all code would be inaccurate.
