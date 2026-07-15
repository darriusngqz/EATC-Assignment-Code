# Windows PE malware classifier (v2)

EATC Assignment 2 project code: a static PE-header malware classifier
(Random Forest / XGBoost / LightGBM, deployed as a Streamlit app) with a
Multi-Layer Perceptron trained alongside for comparison. This is a full
rebuild of the original project on a larger, more representative dataset,
after real-world testing found the original ~19,600-row model flagged
90.5% of genuine Windows system files as malicious. The original project
is fully preserved, untouched, in [`legacy_v1/`](legacy_v1/) (see its own
README for that version's design and findings).

The full modelling workflow lives in `notebooks/`, one notebook per stage
of the data science lifecycle, chained via saved CSVs rather than Python
imports between notebooks. For the full rationale behind each decision,
read the notebooks themselves, they explain each step as they go.

## What changed from v1, and why

`legacy_v1`'s leading suspected cause of its 90.5% real-world false-positive
rate: a ~19,600-row academic CSV whose benign class did not structurally
resemble typical real-world software, combined with only 10 features to
work with. v2 addresses both directly:

- **Dataset**: [EMBER2018](https://github.com/elastic/ember) (Elastic),
  800,000 real-world Windows PE files submitted to VirusTotal 2017-2018,
  exactly balanced (400,000 malicious / 400,000 benign), instead of the
  original ~19,600-row set. See `data/` below for how this project
  re-derives its own feature schema from EMBER's raw JSON rather than
  adopting EMBER's own 2,351-dimension vectorization.
- **Features**: 20 (`src/constants.py`'s `ORDER_OF_FEATURES`), up from 10.
  Keeps the original section-entropy/size stats (validated as genuine
  signal, not a DLL/EXE shortcut, in `legacy_v1/notebooks/03_eda.ipynb`),
  adds import-richness and string-based signals. Still deliberately
  excludes raw header fields (`Machine`, `Characteristics`, `Subsystem`)
  that would let the model shortcut-learn file type instead of malicious
  intent, same anti-shortcut philosophy as v1, not relaxed.
- **Models**: adds LightGBM as a third classical candidate alongside
  Random Forest and XGBoost, since EMBER's own published baseline uses
  LightGBM and reports ROC AUC 0.999 (Anderson & Roth, 2018,
  [arXiv:1804.04637](https://arxiv.org/abs/1804.04637)). `app.py` only
  knows how to load a scikit-learn-style `Pipeline` (`models/classical_pipeline.joblib`),
  same as v1; if the MLP wins in `06_evaluation.ipynb` it is saved separately
  and `app.py` needs extending to serve it, see `06`'s own note on this
  before assuming a classical candidate should win.
- **App**: the NSRL/VirusTotal hash-whitelisting layer `legacy_v1/app.py`
  has (an external, independent verification layer, separate from the
  model itself) is **not** included in v2's `app.py` yet. It was a
  mitigation built specifically for v1's confirmed false-positive problem.
  Whether v2 needs it depends on `07_real_world_validation.ipynb`'s result,
  see Section 5 below.

## Dataset

- `data/dataset_pe_v2_full.csv` - 800,000 rows, 20 numeric features +
  `Name` (SHA-256) + `Malware` (label) + `OriginalSplit` (`train`/`test`,
  EMBER's own official split, preserved so results are directly comparable
  to the published EMBER benchmark and to avoid any leakage from an
  in-house re-split). **Not committed to git** (129MB, regenerable, see
  below), gitignored. This is the only CSV in the project, no separate
  train/validation/test files are saved to disk: `04`, `05`, and `06` each
  independently derive the identical split inline (fixed `RANDOM_STATE`,
  same logic each time), the same reproducibility approach `legacy_v1`
  uses. `02_data_preparation.ipynb` shows this split once for reference.
- Built from Elastic's raw EMBER2018 tar (not included in this repo,
  1.6GB+): `01`/`02` assume `data/dataset_pe_v2_full.csv` already exists.
  To regenerate it from scratch, download EMBER2018 from
  [github.com/elastic/ember](https://github.com/elastic/ember) and run
  `python data_pipeline/build_dataset.py path/to/ember_dataset_2018_2.tar`
  (derives the 20 `ORDER_OF_FEATURES` columns from each record's raw JSON,
  keeps only labeled rows, i.e. drops EMBER's ~200,000 unlabeled `-1` rows).
- `data/demo_sample_v2.csv` - a small (25-row, a few KB) precomputed sample
  used only by `app.py`'s "Try a sample" tab, **committed to git** (unlike
  the full CSV above, small enough that GitHub's 100MB per-file push limit
  is a non-issue). Needed because Streamlit Cloud only has access to files
  actually in the repo, the 129MB full dataset is never present there.
  Regenerate it if needed with:
  ```python
  import pandas as pd
  full = pd.read_csv("data/dataset_pe_v2_full.csv")
  test_only = full[full["OriginalSplit"] == "test"]
  test_only.sample(n=25, random_state=42).reset_index(drop=True).to_csv("data/demo_sample_v2.csv", index=False)
  ```

**Dataset size: kept at the full 800,000 rows, not downsampled.**
Evaluated explicitly in `02_data_preparation.ipynb` rather than assumed:
EMBER's own published baseline trains on the complete labeled corpus, tree
ensembles scale close to linearly and handle this row count in minutes on
ordinary hardware, and shrinking a now much larger, better-sourced dataset
without a real compute reason would risk reintroducing the exact problem
this rebuild exists to fix. A small random subsample is used later, in
`06`, purely for SHAP plot legibility, not as a training-data reduction.

`src/constants.py` (`ORDER_OF_FEATURES`) is the single source of truth for
the 20 feature names and their order; every notebook and the app import
it, so they can never silently drift apart. `src/extract_features.py`
computes them from a live uploaded file (via `pefile`); the data-build
step computes the same 20 values from EMBER's raw JSON, kept logically
identical on purpose.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements-dev.txt   # everything, for the notebooks + testing
# or, once you only need to run the deployed app:
pip install -r requirements.txt
```

`requirements.txt` pins `xgboost==2.0.3` and `shap==0.49.1` together on
purpose, same reason as v1: newer XGBoost (3.x) changed how it stores
`base_score` internally in a way SHAP's `TreeExplainer` cannot parse yet.
`lightgbm` is in `requirements.txt` (not just dev) because `06` may select
it as the final deployed model.

## 2. Run the notebooks (in order)

```bash
jupyter notebook notebooks/
```

Run each one top to bottom (Restart & Run All), in this order:

| Stage | Notebook | What it does |
|-------|----------|---------------|
| Data Understanding | [`01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) | First look at the raw 800,000-row dataset: shape, class balance, `OriginalSplit` provenance, missing values, duplicate hashes |
| Data Preparation | [`02_data_preparation.ipynb`](notebooks/02_data_preparation.ipynb) | Cleaning checks, the dataset-size decision (justified in full, see above), and the train/validation/test split shown for reference (not saved to disk, see Dataset section above) |
| EDA | [`03_eda.ipynb`](notebooks/03_eda.ipynb) | Feature correlation heatmap, class-conditional distributions, and a feature-vs-label leakage/shortcut check (the v2 equivalent of v1's DLL/EXE bias diagnostic) |
| Modelling | [`04_modelling_classical.ipynb`](notebooks/04_modelling_classical.ipynb) | Tunes and validates Random Forest, XGBoost, and LightGBM on the 20-feature schema |
| Modelling | [`05_modelling_mlp.ipynb`](notebooks/05_modelling_mlp.ipynb) | Trains and validates the MLP deep-learning comparison model, same data as `04` |
| Evaluation | [`06_evaluation.ipynb`](notebooks/06_evaluation.ipynb) | Compares all four candidates on validation, selects a winner, runs the one-time held-out test-set check (EMBER's own official test split), explains predictions with SHAP, saves `models/classical_pipeline.joblib` |
| Real-World Validation | [`07_real_world_validation.ipynb`](notebooks/07_real_world_validation.ipynb) | Needs real `.exe`/`.dll` files on disk: scans a folder assumed all-legitimate for false positives, explains any with SHAP. **This result decides whether the NSRL/VirusTotal layer needs to be added back to `app.py`**, see Section 5 |

Only `constants.py`, `extract_features.py`, and `explain.py` remain as
importable `.py` modules in `src/`, same reasoning as v1: training and the
deployed app must always compute features and explanations the exact same
way.

## 3. Run the app locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires `models/classical_pipeline.joblib`
(produced by `06_evaluation.ipynb`). Three tabs: upload a real `.exe`/`.dll`,
try a random sample from EMBER's held-out test set (with its real ground-truth
label shown, unlike v1's unlabeled demo file), or batch-classify a CSV of
already-extracted feature rows. The malicious/benign cutoff is the standard
0.5, see `app.py`'s `MALICIOUS_THRESHOLD` comment.

**No NSRL/VirusTotal hash-whitelisting layer yet.** v1's `app.py` added this
as an independent, external verification layer after real-world testing
confirmed a serious false-positive problem (see `legacy_v1/README.md`
Section 5 for the full design and reasoning). v2 deliberately leaves it out
until `07_real_world_validation.ipynb` shows the same problem persists on
this rebuilt model; adding it pre-emptively would treat it as a default
rather than a targeted fix for a confirmed problem. If `07` does show a
meaningful false-positive rate, the same design (two-way NSRL override,
informational-only VirusTotal ratio, both Upload-tab only) can be ported
over from `legacy_v1/app.py` with minimal changes.

## 4. Run the tests

```bash
pytest tests/ -v
```

`tests/test_extract_features.py` runs without `pefile` or a real PE file
(tests the feature-computation logic against small fake objects, same
approach as `legacy_v1/tests`, adapted to the 20-feature schema).
`tests/test_pipeline.py` needs `scikit-learn` and trains a tiny throwaway
pipeline on synthetic data in under a second; it does not touch the real
dataset or the production model file. Both only depend on `constants.py`
and `extract_features.py`, so they are unaffected by anything the
notebooks do, and are safe to run before or after `01`-`07`.

## 5. Validate against real files (run locally, in `07_real_world_validation.ipynb`)

Same two checks as v1's `07`: scan a folder assumed all-legitimate (e.g.
`C:\Windows\System32`) and report the false-positive rate, then explain any
false positives with SHAP. A third check (scoring known-malicious/known-benign
folders for real accuracy) is left out for the same reason as v1: it needs
real malware samples handled in a properly isolated environment this project
does not have safe access to.

**This is the number that decides whether whitelisting is needed.** Compare
the result against v1's confirmed 90.5% (`legacy_v1/notebooks/07_real_world_validation.ipynb`).
Report whatever is found honestly, whether it's a large improvement, a
partial one, or none, this is the actual test of whether the v2 rebuild
(more, better-sourced data + a richer feature set) worked.

## 6. Deploy to Streamlit Community Cloud

1. Push this repository to GitHub, with `app.py` in the repository root and
   `models/classical_pipeline.joblib` committed (the large data CSVs and
   intermediate candidate models are gitignored, see `.gitignore`, only the
   final deployed model file needs to be in the repo).
2. Go to share.streamlit.io, sign in, "Create app", point it at this repo,
   branch, and `app.py`.
3. `requirements.txt` (not `requirements-dev.txt`) is what Streamlit Cloud
   installs, keep `tensorflow`/`pytest`/`matplotlib`/`seaborn` out of it.
4. Manually re-test all three tabs against the live URL once deployed.

## Project layout

```
EATC-Assignment-main/
  legacy_v1/                     # original project, fully preserved, untouched (see its own README)
  data_pipeline/
    build_dataset.py             # EMBER2018 tar -> data/dataset_pe_v2_full.csv (raw feature extraction, no sampling)
  data/
    dataset_pe_v2_full.csv       # 800,000 rows, gitignored (regenerable from EMBER), the only CSV in the project
  models/                        # saved model artifacts (created by the notebooks)
    classical_pipeline.joblib    # the deployed model (tracked, needed for Streamlit Cloud)
  notebooks/
    01_data_understanding.ipynb
    02_data_preparation.ipynb
    03_eda.ipynb
    04_modelling_classical.ipynb
    05_modelling_mlp.ipynb
    06_evaluation.ipynb
    07_real_world_validation.ipynb   # needs real .exe/.dll files on disk; also decides on whitelisting
  src/
    constants.py                 # ORDER_OF_FEATURES (20 features), label/random-state constants
    extract_features.py          # PE parsing -> feature row (shared by every notebook + the app)
    explain.py                   # SHAP explainability (unchanged from legacy_v1, feature-agnostic)
  tests/
    conftest.py                  # adds src/ to the import path
    test_extract_features.py     # feature-computation logic, no pefile/real PE file needed
    test_pipeline.py             # pipeline mechanics on synthetic data, needs scikit-learn
  app.py                         # Streamlit app (deployed), no whitelisting layer yet, see Section 3
  requirements.txt               # deployed app dependencies only
  requirements-dev.txt           # + notebook/testing/plotting dependencies
```

## What has and hasn't been run

11 of `tests/test_extract_features.py`'s 12 tests pass in an environment
without `pefile` installed (the 12th, which needs `pefile` and asserts a
non-PE upload is rejected, is skipped, not failed, and will run once
`pefile` is installed via `requirements.txt`). `tests/test_pipeline.py`
needs `scikit-learn`, run it locally to confirm.

`01_data_understanding.ipynb`, `02_data_preparation.ipynb`, and
`03_eda.ipynb` have been run end to end (this sandbox has pandas/matplotlib/
seaborn but not scikit-learn/xgboost/lightgbm/tensorflow/shap/pefile, so
everything data-understanding/cleaning/EDA-related is real, executed output;
everything model-training-related is not). Confirmed real findings:
800,000 rows, exactly balanced, zero missing values, zero duplicate hashes,
zero zero-variance features; the reference 85/15/EMBER-test split (shown in
`02`, reproduced inline by `04`/`05`/`06`, never saved to disk) produced
127,500 / 22,500 / 200,000 rows, each exactly 50/50 (training pool
downsampled to 150,000 rows, 75,000 per class, before this split, a
disclosed compute-constrained tradeoff added 2026-07 so `GridSearchCV`
finishes on ordinary laptop hardware; the held-out 200,000-row EMBER test
split is untouched, see the note above the split cell in `02`/`04`/`05`/`06`);
two feature pairs correlated
above 0.85 (`SectionMaxVirtualsize`/`VirtualSize` at 0.935,
`NumStrings`/`FileSize` at 0.897, both kept, tree models are not harmed by
correlated inputs); no single feature approaches |r| = 1.0 with the label
(strongest is `SectionMaxEntropy` at 0.37), confirming no shortcut feature
snuck back into the widened feature set.

`04_modelling_classical.ipynb`, `05_modelling_mlp.ipynb`, and
`06_evaluation.ipynb` need a fresh **Restart & Run All** on a machine with
`scikit-learn`/`xgboost`/`lightgbm`/`tensorflow`/`shap` installed before
their accuracy/F1/ROC-AUC numbers and the model files they save
(`models/rf_v2.joblib`, `models/xgb_v2.joblib`, `models/lgbm_v2.joblib`,
`models/mlp_v2.keras`, `models/classical_pipeline.joblib`) can be trusted.
Each notebook's Summary cell has a `TODO` marking exactly what to fill in.

`07_real_world_validation.ipynb` needs `06`'s model file plus real local
`.exe`/`.dll` files, and has not been run yet. Its result is the actual
test of whether this rebuild fixed v1's 90.5% false-positive problem, and
decides whether the NSRL/VirusTotal layer needs to be ported over from
`legacy_v1/app.py`, see Section 3 above. Report whatever is found honestly.
