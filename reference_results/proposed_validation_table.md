# Proposed Validation

Three nested outer validation folds inside seed 42's 7,200 training rows. Original test rows are excluded. Mean ± fold SD, not confidence intervals. Original SVM has 12 settings; calibrated search has 24 plus calibration fits. Exploratory findings; do not combine these scores with the five-holdout baseline table.

| Model | Macro-F1 | Accuracy | Balanced accuracy | Acute precision | Acute recall |
|---|---:|---:|---:|---:|---:|
| RBF-SVM | 0.8462 ± 0.0152 | 0.8750 ± 0.0014 | 0.8273 ± 0.0201 | 0.8429 ± 0.0458 | 0.7024 ± 0.0743 |
| Calibrated SVM + decision rule | 0.8492 ± 0.0134 | 0.8764 ± 0.0027 | 0.8290 ± 0.0134 | 0.8551 ± 0.0502 | 0.7024 ± 0.0412 |
