# Verification checks

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

- `test_histogram.py`: lecture examples, probabilities, and loss-based decisions.
- `test_proposed.py`: calibration-fold scaling and acute-class decision behavior.
- `test_reproduction.py`: data integrity, baseline search budgets, generated
  evaluation evidence, test-row exclusion, and source-package contents.

Run `python reproduce.py --download-data --proposed` first to generate the
submission outputs used by the evidence checks; those checks otherwise skip.
