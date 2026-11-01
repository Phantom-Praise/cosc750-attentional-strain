# Baseline

Five shared stratified 80/20 holdouts; mean ± sample SD. All learned models use the same six features and 12 training-only CV settings. Holdouts overlap; SD is not a confidence interval. Histogram NB was introduced after original baseline results were viewed. Synthetic-data results, not clinical validation.

| Model | Macro-F1 | Accuracy | Balanced accuracy | Acute precision | Acute recall |
|---|---:|---:|---:|---:|---:|
| Majority reference | 0.1502 ± 0.0000 | 0.4294 ± 0.0000 | 0.2500 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Histogram Naive Bayes | 0.6765 ± 0.0128 | 0.7621 ± 0.0042 | 0.7166 ± 0.0320 | 0.3927 ± 0.0228 | 0.7143 ± 0.1468 |
| Logistic regression | 0.8327 ± 0.0228 | 0.8694 ± 0.0067 | 0.8142 ± 0.0256 | 0.8222 ± 0.0763 | 0.6571 ± 0.1032 |
| Random forest | 0.8291 ± 0.0248 | 0.8654 ± 0.0081 | 0.8160 ± 0.0356 | 0.7966 ± 0.0668 | 0.6762 ± 0.1278 |
| RBF-SVM | 0.8387 ± 0.0207 | 0.8704 ± 0.0087 | 0.8181 ± 0.0224 | 0.8403 ± 0.0974 | 0.6857 ± 0.0865 |
