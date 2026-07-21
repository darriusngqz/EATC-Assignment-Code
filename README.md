# Windows PE malware classifier

EATC Assignment 2 project code: a static PE-header malware classifier (Random Forest / XGBoost, deployed as a Streamlit app) with a Multi-Layer Perceptron trained alongside for comparison. Trained on [EMBER2018](https://github.com/elastic/ember) (Elastic), 800,000 real-world Windows PE files submitted to VirusTotal 2017-2018, exactly balanced (400,000 malicious / 400,000 benign).

The full modelling workflow lives in `notebooks/`, one notebook per stage of the data science lifecycle, chained via saved CSVs rather than Python imports between notebooks. For the full rationale behind each decision, read the notebooks themselves, they explain each step as they go.

## Design

- **Dataset**: EMBER2018, 800,000 rows. This project re-derives its own 20-feature schema from EMBER's raw JSON rather than adopting EMBER's own 2,351-dimension vectorization, see `data/` below.
- **Features**: 20 (`src/constants.py`'s `ORDER_OF_FEATURES`): section-table entropy/size stats, import richness, string-based signals, and overall file size. Deliberately excludes raw header fields (`Machine`, `Characteristics`, `Subsystem`) that would let the model shortcut-learn file type instead of malicious intent, an anti-shortcut check is run explicitly in `03_eda.ipynb`.
- **Models**: Random Forest and XGBoost (each a scikit-learn `Pipeline` with `StandardScaler`), plus a Multi-Layer Perceptron trained as a deep-learning comparison point. `app.py` only knows how to load a scikit-learn-style `Pipeline` (`models/classical_pipeline.joblib`); the MLP is comparison-only and is not a deployment candidate.
- **App**: three tabs (upload a real file, try a random labelled sample, batch-classify a CSV), backed directly by the deployed model's own prediction, see Section 5 below for its real-world false-positive rate.

## Dataset

- `data/dataset_pe_v2_full.csv` - 800,000 rows, 20 numeric features + `Name` (SHA-256) + `Malware` (label) + `OriginalSplit` (`train`/`test`, EMBER's own official split, preserved so results are directly comparable to the published EMBER benchmark and to avoid any leakage from an in-house re-split). **Not committed to git** (129MB, regenerable, see below), gitignored. This is the only full CSV in the project, no separate train/validation/test files are saved to disk: `04`, `05`, and `06` each independently derive the identical split inline (fixed `RANDOM_STATE`, same logic each time). `02_data_preparation.ipynb` shows this split once for reference.
- Built from Elastic's raw EMBER2018 tar (not included in this repo, 1.6GB+): `01`/`02` assume `data/dataset_pe_v2_full.csv` already exists. To regenerate it from scratch, download EMBER2018 from [github.com/elastic/ember](https://github.com/elastic/ember) and run `python data_pipeline/build_dataset.py path/to/ember_dataset_2018_2.tar` (derives the 20 `ORDER_OF_FEATURES` columns from each record's raw JSON, keeps only labeled rows, i.e. drops EMBER's ~200,000 unlabeled `-1` rows).
- `data/demo_sample_v2.csv` - a small (25-row, a few KB) precomputed sample used only by `app.py`'s "Try a sample" tab, **committed to git**. Needed because Streamlit Cloud only has access to files actually in the repo, the 129MB full dataset is never present there. Regenerate it if needed with:
  ```python
  import pandas as pd
  full = pd.read_csv("data/dataset_pe_v2_full.csv")
  test_only = full[full["OriginalSplit"] == "test"]
  test_only.sample(n=25, random_state=42).reset_index(drop=True).to_csv("data/demo_sample_v2.csv", index=False)
  ```

**Dataset size: kept at the full 800,000 rows, not downsampled.** Evaluated explicitly in `02_data_preparation.ipynb` rather than assumed: EMBER's own published baseline trains on the complete labeled corpus, and tree ensembles scale close to linearly and handle this row count in minutes on ordinary hardware. A small random subsample is used later, in `06`, purely for SHAP plot legibility, not as a training-data reduction.

**Training-pool downsample (compute, not dataset-size, decision):** `GridSearchCV` over the full 600,000-row training pool proved too slow to iterate on with the available hardware, so `02`, `04`, `05`, and `06` downsample the training pool to 150,000 rows (`TRAIN_POOL_SAMPLE_PER_CLASS = 75000`, stratified, fixed `RANDOM_STATE`) before the 85/15 train/validation split, producing 127,500 train / 22,500 validation rows. **The held-out 200,000-row EMBER test split is never touched or reduced** by this, `06`'s final reported test metrics remain on the full, standard benchmark split. See the rationale cell immediately above the split cell in each notebook.

`src/constants.py` (`ORDER_OF_FEATURES`) is the single source of truth for the 20 feature names and their order; every notebook and the app import it, so they can never silently drift apart. `src/extract_features.py` computes them from a live uploaded file (via `pefile`); `data_pipeline/build_dataset.py` computes the same 20 values from EMBER's raw JSON, kept logically identical on purpose.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements-dev.txt   # everything, for the notebooks + testing
# or, once you only need to run the deployed app:
pip install -r requirements.txt
```

`requirements.txt` pins `xgboost==2.0.3` and `shap==0.49.1` together on purpose: newer XGBoost (3.x) changed how it stores `base_score` internally in a way SHAP's `TreeExplainer` cannot parse yet.

## 2. Run the notebooks (in order)

```bash
jupyter notebook notebooks/
```

Run each one top to bottom (Restart & Run All), in this order:

| Stage | Notebook | What it does |
|-------|----------|---------------|
| Data Understanding | [`01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) | First look at the raw 800,000-row dataset: shape, class balance, `OriginalSplit` provenance, missing values, duplicate hashes |
| Data Preparation | [`02_data_preparation.ipynb`](notebooks/02_data_preparation.ipynb) | Cleaning checks, the dataset-size decision (justified in full, see above), and the train/validation/test split shown for reference (not saved to disk, see Dataset section above) |
| EDA | [`03_eda.ipynb`](notebooks/03_eda.ipynb) | Feature correlation heatmap, class-conditional distributions, and a feature-vs-label leakage/shortcut check |
| Modelling | [`04_modelling_classical.ipynb`](notebooks/04_modelling_classical.ipynb) | Tunes and validates Random Forest and XGBoost on the 20-feature schema via `GridSearchCV` |
| Modelling | [`05_modelling_mlp.ipynb`](notebooks/05_modelling_mlp.ipynb) | Trains and validates the MLP deep-learning comparison model, same data as `04` |
| Evaluation | [`06_evaluation.ipynb`](notebooks/06_evaluation.ipynb) | Compares all three candidates on validation, selects a winner, runs the one-time held-out test-set check (EMBER's own official test split), explains predictions with SHAP, saves `models/classical_pipeline.joblib` |
| Real-World Validation | [`07_real_world_validation.ipynb`](notebooks/07_real_world_validation.ipynb) | Needs real `.exe`/`.dll` files on disk: scans a folder assumed all-legitimate (e.g. `C:\Windows\System32`) for false positives, explains any with SHAP |

Only `constants.py`, `extract_features.py`, and `explain.py` remain as importable `.py` modules in `src/`: training and the deployed app must always compute features and explanations the exact same way.

**Confirmed results (from a real, executed run):**

| Model | Validation accuracy | Validation F1 | Validation ROC-AUC |
|-------|---------------------|----------------|---------------------|
| Random Forest | 95.14% | 0.9514 | 0.9902 |
| XGBoost | 94.70% | 0.9470 | 0.9887 |
| MLP (comparison only) | 88.57% | 0.8857 | 0.9559 |

Random Forest scores marginally best on validation, but **XGBoost is the deployed model**: Random Forest's saved model file is 459MB versus XGBoost's 3.7MB, and GitHub rejects any single file over 100MB, so Random Forest cannot be committed for a Streamlit Cloud deployment. This is a disclosed deployability tradeoff, not a claim that XGBoost won on raw validation numbers.

On the untouched 200,000-row EMBER test split, the deployed XGBoost model scores **93.0% accuracy, ROC-AUC 0.979** (confusion matrix: 92,983 true negatives, 7,017 false positives, 7,232 false negatives, 92,768 true positives).

## 3. Run the app locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires `models/classical_pipeline.joblib` (produced by `06_evaluation.ipynb`). Three tabs: upload a real `.exe`/`.dll`, try a random sample from EMBER's held-out test set (with its real ground-truth label shown), or batch-classify a CSV of already-extracted feature rows. The malicious/benign cutoff is the standard 0.5, see `app.py`'s `MALICIOUS_THRESHOLD` comment. Upload limit is 200MB, set in both `app.py` (`MAX_UPLOAD_MB`) and `.streamlit/config.toml` (`server.maxUploadSize`), kept in sync since Streamlit's upload widget displays and enforces its own config-level limit independently of the app's own code.

## 4. Run the tests

```bash
pytest tests/ -v
```

`tests/test_extract_features.py` (12 tests) runs without `pefile` or a real PE file, testing the feature-computation logic against small hand-built fake objects. `tests/test_pipeline.py` (4 tests) needs `scikit-learn` and trains a tiny throwaway pipeline on synthetic data in under a second; it does not touch the real dataset or the production model file. Both only depend on `constants.py` and `extract_features.py`, so they are unaffected by anything the notebooks do, and are safe to run before or after `01`-`07`.

## 5. Validate against real files (`07_real_world_validation.ipynb`)

Scans a folder assumed all-legitimate (200 files under `C:\Windows\System32`) and reports the false-positive rate, then explains any false positives with SHAP.

**Confirmed result: 1.5% false-positive rate (3 out of 200 files).** All three false positives were traced via SHAP to elevated `SectionMaxEntropy` and unusually low `SectionMinRawsize` (4096 bytes, versus a benign average around 42,850), a genuine, explainable edge case rather than a systemic problem. Note this number is lower than the EMBER test-set false-positive rate (7.0%, from the confusion matrix above): the two measure different populations, EMBER's test set is a harder, VirusTotal-sourced benchmark, System32 is a narrower, typical-legitimate-software population, so both numbers are reported rather than picking the more flattering one.

A second check (scoring known-malicious/known-benign folders for a full accuracy readout) is left out: it needs real malware samples handled in a properly isolated environment this project does not have safe access to.

## 6. Deploy to Streamlit Community Cloud

1. Push this repository to GitHub, with `app.py` in the repository root and `models/classical_pipeline.joblib` committed (the large data CSVs and intermediate candidate models are gitignored, see `.gitignore`, only the final deployed model file needs to be in the repo).
2. Go to share.streamlit.io, sign in, "Create app", point it at this repo, branch, and `app.py`.
3. `requirements.txt` (not `requirements-dev.txt`) is what Streamlit Cloud installs, keep `tensorflow`/`pytest`/`matplotlib`/`seaborn` out of it.
4. `.streamlit/config.toml` sets `server.maxUploadSize` to match `app.py`'s own `MAX_UPLOAD_MB` (200MB), without it the upload widget silently shows Streamlit's own default limit instead of what the app actually enforces.
5. Manually re-test all three tabs against the live URL once deployed, the demo tab and the upload-limit widget both behave differently on the cloud platform than locally.

## Project layout

```
EATC-Assignment-Code/
  data_pipeline/
    build_dataset.py             # EMBER2018 tar -> data/dataset_pe_v2_full.csv (raw feature extraction, no sampling)
  data/
    dataset_pe_v2_full.csv       # 800,000 rows, gitignored (regenerable from EMBER)
    demo_sample_v2.csv           # 25-row committed sample, powers the app's Try a Sample tab
  models/
    classical_pipeline.joblib    # the deployed model (tracked, needed for Streamlit Cloud)
  notebooks/
    01_data_understanding.ipynb
    02_data_preparation.ipynb
    03_eda.ipynb
    04_modelling_classical.ipynb
    05_modelling_mlp.ipynb
    06_evaluation.ipynb
    07_real_world_validation.ipynb
  src/
    constants.py                 # ORDER_OF_FEATURES (20 features), label/random-state constants
    extract_features.py          # PE parsing -> feature row (shared by every notebook + the app)
    explain.py                   # SHAP explainability
  tests/
    conftest.py                  # adds src/ to the import path
    test_extract_features.py     # feature-computation logic, no pefile/real PE file needed
    test_pipeline.py             # pipeline mechanics on synthetic data, needs scikit-learn
  .streamlit/
    config.toml                  # upload-size limit, kept in sync with app.py's MAX_UPLOAD_MB
  app.py                         # Streamlit app (deployed)
  requirements.txt               # deployed app dependencies only
  requirements-dev.txt           # + notebook/testing/plotting dependencies
```
