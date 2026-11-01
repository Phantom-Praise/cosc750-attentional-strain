# Running the project

Use Python 3.12 and install the pinned requirements from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --download-data
```

Windows activation: `.venv\Scripts\activate`.

The download flag obtains Kaggle version 1 only if the CSV is absent, verifying
the expected SHA256 before writing. For your own copy:

```bash
python reproduce.py --data /path/to/short_form_brain_rot_cognitive_decay.csv
```

The five baselines share six inputs, five stratified 80/20 splits, and training-
only three-fold tuning. Each learned model gets 12 configurations. No earlier
result files or lecture PDFs are needed.

## Optional proposed comparison

```bash
python reproduce.py --download-data --proposed --output results/full_run
```

This also runs a separate nested-validation comparison within seed 42's training
partition. The original test rows stay excluded. Do not mix its validation scores
with the baseline test scores. The original SVM has 12 settings; the calibrated
search has 24 settings and additional calibration fits.

## Outputs

By default, files are written to `results/submission/`:

- `baseline_table.md`, `.csv`, `.tex`: the main comparison.
- `baseline_comparison.png`, `.svg`: preview and vector chart.
- `baseline_per_run.csv`, `baseline_per_class.csv`: detailed metrics and errors.
- `baseline/`: saved splits, predictions, settings, and CV evidence.
- `manifest.json`: completion status, versions, source/data hashes, and hardware.

With `--proposed`, corresponding `proposed_validation_*` files and fold evidence
are added. The LaTeX tables use `booktabs`. Search time includes refit time, so
do not add the separately reported refit time twice.

Existing results are protected. Use a new `--output results/repeat_01` directory
for another run. `--jobs 2` reduces parallel search workers.

## Tests and code ZIP

```bash
python -m unittest discover -s tests -v
python tools/build_submission.py
```

Tests requiring generated submission outputs skip until reproduction has run.
The ZIP contains the focused workflow, tests, documentation and reference tables.
It excludes the dataset, environment, archive, local outputs, and model binaries.
Recorded reference tables are never used to train or manufacture new results.
