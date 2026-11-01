# Submission code packaging

From the repository root, run:

```bash
python tools/build_submission.py
```

`build_submission.py` creates `dist/cosc750-stage3-code.zip` with the main
workflow, tests, documentation, reference results, and a file-checksum manifest.
It excludes the dataset, virtual environment, generated runs, fitted models,
Git internals, and archived experiments.

This packages the code only. The cumulative paper, slides, and recording belong
in the final assignment submission bundle.
