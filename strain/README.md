# Main code

| File | What to look for |
|---|---|
| [config.py](config.py) | Six features, four labels, dataset checks, baseline definitions |
| [histogram.py](histogram.py) | Histogram Naive Bayes and loss-based decisions |
| [proposed.py](proposed.py) | Calibrated SVM and acute-class decision rule |
| [workflow.py](workflow.py) | Splits, training, tuning, evaluation, and report export |

Start with `config.py`, then follow `main()` in `workflow.py`. The root
`reproduce.py` is a small entry point to this package.

The baseline definitions are in `config.py`; custom classifiers are in
`histogram.py` and `proposed.py`. Training and evaluation are in `workflow.py`.
