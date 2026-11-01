# Submission-code verification

Verified November 1, 2026 on this project's Apple M3 Pro, macOS ARM64 machine
using Python 3.12 and the pinned requirements. Other operating systems were not
executed in this check.

## End-to-end workflow

1. Ran `python reproduce.py --proposed` in the workspace.
2. Compared all 25 baseline prediction sets with the original experiments, and
   all six ordinary/calibrated outer-fold prediction sets with earlier nested
   validation. Every predicted label and evaluated row matched.
3. Built the source ZIP and checked every packaged file against its manifest.
4. Extracted into a new temporary directory with no dataset, earlier results,
   environment, or lecture PDFs.
5. Created a new virtual environment and installed `requirements.txt` from scratch.
6. Ran `python reproduce.py --download-data --proposed` there. The public version-1
   download passed the expected dataset checksum.
7. Compared all 31 newly generated prediction CSVs and every predictive summary
   value with the workspace reproduction. They matched exactly. Timing differed
   as expected and was not used for this equality check.

## Checks

- Workspace: all 30 tests passed.
- Clean folder: 25 tests passed, five historical-evidence tests skipped because
  the older experiment directories were intentionally absent. Both new
  submission-evidence tests ran, including proposed-method test-row exclusion.
- After source-packaging changes, all seven reproduction/packaging tests passed.
- Generated figures were visually inspected. Tables are also exported as CSV,
  Markdown, and booktabs LaTeX source.
- Source-package checks exclude the virtual environment, dataset CSV, results
  debris, local Git internals, and fitted model binaries.
- Existing experimental source files and their original evidence remain intact.

The one-command baseline workflow retrains models; it does not copy recorded
reference scores. Optional proposed validation also regenerates its folds and
fits from data. Earlier exploratory decisions and repeated use of the same
benchmark are still disclosed; reproduction does not create new independent
statistical evidence.

## Remaining submission boundaries

The hosted repository is private by default; grant the instructor access before
submitting its link. No collaborators were invited or messaged automatically.
The source ZIP is a code artifact, not the final paper/slides/recording ZIP.
The publisher's conflicting dataset license fields are recorded in DATASET.md;
the cumulative paper needs that correction when it is updated.
