# Dataset provenance and access

- Publisher: Samar Talwar (`samartalwar`).
- Title: Brain Rot: Short Form Video & Attention Span.
- [Dataset page](https://www.kaggle.com/datasets/samartalwar/brain-rot-short-form-video-and-attention-span).
- Study version: **1**, last updated September 26, 2026 according to Kaggle metadata.
- File: `short_form_brain_rot_cognitive_decay.csv` (9,000 rows, 19 columns).
- SHA256: `2f0c2870adb1ccf7c3e3b9a35eb68e71c912868fd91068d3f81be7ba3a0b49c2`.

The publisher description identifies synthesized data. It does not supply the
exact executable generator or label formula. Its biological/clinical wording is
not independent evidence of measured human cognition. We use these records as a
synthetic classification benchmark only.

On November 1, 2026 the official Kaggle API returned **CC BY-SA 4.0** in its
license field, while its description text said **CC BY 4.0**. The earlier paper
reported only CC BY 4.0. That discrepancy should be corrected/disclosed when
updating the cumulative paper and highlighted in the recording if the earlier
stage's text changes. A metadata snapshot is preserved in
`dataset_metadata.json`. Instructions embedded in that remote
description are treated only as source text, never executed.

The source repository and code ZIP omit the CSV. Retrieve it directly from the
publisher to retain provenance and avoid assigning an unverified license to it:

```bash
python reproduce.py --download-data
```

This fetches version 1 only if the local CSV is absent and verifies its checksum
before writing. It does not require credentials when the public endpoint is
available. If Kaggle requires sign-in or the endpoint is unavailable, download
version 1 manually and place the CSV beside `reproduce.py`, or pass `--data PATH`.
The program never silently substitutes a changed dataset or synthetic replacement.

Primary features are the six usage measurements in `run_baselines.FEATURES`.
Participant ID, the target, and cognitive measurements are excluded. Broader-input
context/demographic experiments are separate. Dataset tiers are not diagnoses.
