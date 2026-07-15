"""Windows PE file malware checker - Streamlit app (v2).

Security boundary: only ever parses a file's static header structure via
pefile. Never saves an upload to an executable location, never calls
os.system/subprocess/exec, never runs or unpacks bytes beyond pefile's own
header parsing. See src/extract_features.py.

Label convention: predicts P(malicious). Malware == 1 malicious, 0 benign.

v2 vs legacy_v1/app.py: trained on EMBER2018 (800,000 real-world files,
see notebooks/01-02) instead of the original ~19,600-row academic CSV, and
uses 20 features instead of 10 (src/constants.py). Deliberately does NOT
include the NSRL/VirusTotal hash-whitelisting layer that legacy_v1/app.py
has: that layer was a mitigation for v1's confirmed ~90% real-world false
positive rate. Whether v2 needs it depends on notebooks/07's real-world
scan result, add it back only if that shows the same problem persists.

Run locally with: streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
import pandas as pd
import streamlit as st

from explain import build_explainer, explain_prediction
from extract_features import extract_pe_features

MAX_UPLOAD_MB = 200
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classical_pipeline.joblib")
# data/dataset_pe_v2_full.csv (129MB) is gitignored, too large for a normal
# git push (GitHub rejects any single file over 100MB) and never present on
# Streamlit Cloud's clone of the repo. The demo tab instead loads a small,
# precomputed 25-row sample (data/demo_sample_v2.csv, a few KB, committed to
# git) drawn once from the full CSV's OriginalSplit == "test" rows with the
# same random_state=42 used everywhere else in this project. Regenerate it
# with: pandas.read_csv(full_csv) -> filter test rows -> .sample(n=25,
# random_state=42) -> .to_csv(), see README.md.
DEMO_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "demo_sample_v2.csv")
DEMO_SAMPLE_SIZE = 25

# Standard 0.5 cutoff, matches legacy_v1's convention. Real-world reliability
# is measured and reported honestly in notebooks/07_real_world_validation.ipynb.
MALICIOUS_THRESHOLD = 0.5


# Loads the trained model once and caches it. Needed because Streamlit
# reruns this whole script on every click; without caching the model would
# reload from disk every single time.
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    feature_order = artifact["feature_order"]
    explainer = build_explainer(pipeline)
    return pipeline, feature_order, explainer


# Loads a small random sample of the held-out EMBER test set for the "Try a
# sample" tab, cached so the same sample persists across reruns within one
# session (st.cache_data keys on the function + args, a fixed random_state
# makes this reproducible across sessions too).
@st.cache_data
def load_demo_sample():
    if not os.path.exists(DEMO_DATA_PATH):
        return None
    return pd.read_csv(DEMO_DATA_PATH)


# Shows the malicious/benign verdict, confidence, and SHAP explanation for
# one row of features. Shared by the Upload and Try a Sample tabs so both
# display results identically. true_label is optional (only known for the
# demo sample, which comes from a labeled dataset, unlike a live upload).
def render_verdict(pipeline, explainer, feature_order, row_df, true_label=None):
    proba_malicious = pipeline.predict_proba(row_df)[0][1]
    verdict = "Likely malicious" if proba_malicious >= MALICIOUS_THRESHOLD else "Likely benign"

    if verdict == "Likely malicious":
        st.error(f"{verdict} - confidence {proba_malicious:.0%}")
    else:
        st.success(f"{verdict} - confidence {1 - proba_malicious:.0%}")

    if true_label is not None:
        true_text = "malicious" if true_label == 1 else "benign"
        predicted_correct = (verdict == "Likely malicious") == (true_label == 1)
        if predicted_correct:
            st.caption(f"Ground truth (EMBER label): {true_text} - model agrees.")
        else:
            st.caption(f"Ground truth (EMBER label): {true_text} - model disagrees.")

    toward_mal, toward_benign = explain_prediction(explainer, pipeline, row_df.values, feature_order)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Pushing toward malicious**")
        for name, _ in toward_mal:
            st.write(f"- {name}")
    with col2:
        st.write("**Pushing toward benign**")
        for name, _ in toward_benign:
            st.write(f"- {name}")


# Builds the page: title, model load check, and the three tabs. Entry
# point for `streamlit run app.py`.
def main():
    st.set_page_config(page_title="PE malware checker", page_icon=":shield:")
    st.title("Windows PE file malware checker")
    st.write(
        "Static structural analysis of Windows executables (.exe / .dll). "
        "Files are parsed, never executed. This is a coursework prototype, "
        "not a production security tool, do not rely on it for real "
        "protection decisions."
    )

    pipeline, feature_order, explainer = load_pipeline()
    if pipeline is None:
        st.warning(
            f"No trained model found at `{MODEL_PATH}`. Run notebooks 01-06 in "
            "`notebooks/` first (Restart & Run All, in order), see README.md. "
            "07_real_world_validation.ipynb is optional and does not produce this file."
        )
        return

    tab_upload, tab_sample, tab_batch = st.tabs(["Upload a file", "Try a sample", "Batch CSV"])

    with tab_upload:
        # Upload a real .exe/.dll and get a live verdict.
        uploaded = st.file_uploader("PE file", type=["exe", "dll"])
        if uploaded is not None:
            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File too large (limit {MAX_UPLOAD_MB} MB).")
            else:
                file_bytes = uploaded.read()
                try:
                    # Raises on anything that isn't a real, parseable PE
                    # file; caught so a bad upload shows a message, not a crash.
                    row = extract_pe_features(file_bytes, feature_order)
                except Exception:
                    st.error(
                        "This does not look like a valid PE file, or it "
                        "uses a structure this parser cannot read."
                    )
                else:
                    row_df = pd.DataFrame([row], columns=feature_order)
                    render_verdict(pipeline, explainer, feature_order, row_df)

    with tab_sample:
        # Pick from a random sample of EMBER's own held-out test set, real
        # files (by hash) with a genuine, known ground-truth label, useful
        # for a live demo without needing to source an executable yourself.
        st.write(
            f"{DEMO_SAMPLE_SIZE} real files randomly drawn from EMBER2018's held-out "
            "test set (identified by SHA-256 hash, not filename, EMBER does not "
            "ship original filenames). Unlike legacy_v1's demo file, these rows "
            "have a genuine ground-truth label attached, so the verdict below can "
            "be checked directly against it."
        )
        demo = load_demo_sample()
        if demo is None:
            st.info(
                f"No demo data found at `{DEMO_DATA_PATH}`. This small file "
                "should be committed to the repo, see README.md for how to "
                "regenerate it from data/dataset_pe_v2_full.csv if missing."
            )
        else:
            choice = st.selectbox(
                "Pick a file (by hash)", options=list(range(len(demo))),
                format_func=lambda i: f"{demo['Name'].iloc[i][:16]}... "
                                       f"({'malicious' if demo['Malware'].iloc[i] == 1 else 'benign'} per EMBER)"
            )
            row_df = demo.loc[[choice], feature_order]
            true_label = int(demo.loc[choice, "Malware"])
            render_verdict(pipeline, explainer, feature_order, row_df, true_label=true_label)

    with tab_batch:
        # Upload a CSV of many already-extracted feature rows and classify
        # them all in one go.
        st.write(
            f"Upload a CSV of already-extracted PE header rows (the same "
            f"{len(feature_order)} columns this deployed model was trained on: "
            f"{', '.join(feature_order)}) to classify many files at once."
        )
        batch_file = st.file_uploader("CSV of feature rows", type=["csv"], key="batch")
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
            except Exception:
                # Not valid CSV at all (unreadable, wrong format).
                st.error("This does not look like a valid CSV file.")
            else:
                missing = [c for c in feature_order if c not in batch_df.columns]
                if missing:
                    st.error(f"CSV is missing expected columns: {missing}")
                else:
                    X_batch = batch_df[feature_order]
                    try:
                        # Valid CSV, right columns, but bad values inside
                        # (blanks, text instead of numbers).
                        proba = pipeline.predict_proba(X_batch)[:, 1]
                    except Exception:
                        st.error(
                            "Could not score this CSV. Check every row has a "
                            "numeric value in every expected column (no blanks "
                            "or text), then try again."
                        )
                    else:
                        results = batch_df.copy()
                        results["verdict"] = ["malicious" if p >= MALICIOUS_THRESHOLD else "benign" for p in proba]
                        results["confidence"] = [p if v == "malicious" else 1 - p
                                                  for p, v in zip(proba, results["verdict"])]
                        st.dataframe(results)
                        st.download_button(
                            "Download results as CSV",
                            results.to_csv(index=False).encode("utf-8"),
                            file_name="batch_results.csv",
                        )


if __name__ == "__main__":
    main()