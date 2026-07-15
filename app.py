"""Windows PE file malware checker - Streamlit app.

Security boundary: only ever parses a file's static header structure via
pefile. Never saves an upload to an executable location, never calls
os.system/subprocess/exec, never runs or unpacks bytes beyond pefile's own
header parsing. See src/extract_features.py.

Label convention: predicts P(malicious). Malware == 1 malicious, 0 benign.
Run locally with: streamlit run app.py
"""
import sys
import os
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
import pandas as pd
import requests
import streamlit as st

from explain import build_explainer, explain_prediction
from extract_features import extract_pe_features

MAX_UPLOAD_MB = 20
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classical_pipeline.joblib")
DEMO_FILES_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset_test.csv")

# Standard 0.5 cutoff. This model trains on the original, unaugmented
# dataset_malwares.csv (see 06_evaluation.ipynb): no real or synthetic data is
# added to the training set to influence any evaluation result. Real-world
# false-positive rate against C:\Windows\System32 is measured and reported
# honestly in 07_real_world_validation.ipynb and README.md Section 5, as a
# dataset shift finding, not something engineered away by reshaping the
# training data.
MALICIOUS_THRESHOLD = 0.5

# CIRCL hashlookup (hashlookup.circl.lu) is a free public API run by
# Luxembourg's national CERT. It queries NIST's National Software Reference
# Library (NSRL) and a few other legitimate software sources live, no local
# copy of the database is stored or shipped with this project. This is a
# separate, independent verification layer: it does not touch the training
# data, features, or model in any way, it only checks an uploaded file's
# SHA-256 hash against an external, independently maintained list of
# known-legitimate files. It is best-effort with no uptime guarantee, so a
# failed request must fall back to the model's own verdict, never crash the
# app. This only works on the Upload tab, where the original file's bytes
# exist. The Try a Sample and Batch CSV tabs only ever have pre-extracted
# numeric features, there is no original file left to hash.
HASHLOOKUP_URL = "https://hashlookup.circl.lu/lookup/sha256/"
HASHLOOKUP_TIMEOUT = 5


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


# Loads and caches the demo CSV, same reasoning as load_pipeline but for
# data (@st.cache_data is the correct decorator for data, not a model).
@st.cache_data
def load_demo_files():
    if not os.path.exists(DEMO_FILES_PATH):
        return None
    return pd.read_csv(DEMO_FILES_PATH)


# Looks up a file's SHA-256 against CIRCL's hashlookup service. Returns the
# matched file's name on a hit, None on a confirmed no-match (HTTP 404), or
# the string "unavailable" if the service could not be reached, so callers
# can tell "checked, no match" apart from "couldn't check."
def check_nsrl_hash(file_bytes):
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    try:
        response = requests.get(f"{HASHLOOKUP_URL}{sha256}", timeout=HASHLOOKUP_TIMEOUT)
    except requests.RequestException:
        return "unavailable"
    if response.status_code == 200:
        return response.json().get("FileName", "known file")
    if response.status_code == 404:
        return None
    return "unavailable"


# Shows the malicious/benign verdict, confidence, and SHAP explanation for
# one row of features. Shared by the Upload and Try a Sample tabs so both
# display results identically. proba_malicious can be passed in if already
# computed, to avoid scoring the same row twice.
def render_verdict(pipeline, explainer, feature_order, row_df, proba_malicious=None):
    if proba_malicious is None:
        proba_malicious = pipeline.predict_proba(row_df)[0][1]
    verdict = "Likely malicious" if proba_malicious >= MALICIOUS_THRESHOLD else "Likely benign"

    if verdict == "Likely malicious":
        st.error(f"{verdict} - confidence {proba_malicious:.0%}")
    else:
        st.success(f"{verdict} - confidence {1 - proba_malicious:.0%}")

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
                    proba_malicious = pipeline.predict_proba(row_df)[0][1]
                    model_verdict = (
                        "Likely malicious" if proba_malicious >= MALICIOUS_THRESHOLD
                        else "Likely benign"
                    )
                    nsrl_match = check_nsrl_hash(file_bytes)

                    if nsrl_match and nsrl_match != "unavailable":
                        st.success(f"Likely benign - verified via NSRL (matched known file: {nsrl_match})")
                        st.caption(
                            "This file's SHA-256 hash matched a known-legitimate entry "
                            "in NIST's National Software Reference Library (via CIRCL "
                            "hashlookup), an independent reference database this "
                            "project did not build or curate. This overrides the "
                            "model's own verdict for this file."
                        )
                        model_confidence = (
                            proba_malicious if model_verdict == "Likely malicious"
                            else 1 - proba_malicious
                        )
                        st.caption(
                            f"Model's own prediction, shown for transparency: "
                            f"{model_verdict} - confidence {model_confidence:.0%}"
                        )
                    else:
                        if nsrl_match == "unavailable":
                            st.caption(
                                "NSRL reference check unavailable right now "
                                "(request failed or timed out). Showing the model's "
                                "own verdict only."
                            )
                        else:
                            st.caption("No match found in the NSRL reference database.")
                        render_verdict(pipeline, explainer, feature_order, row_df, proba_malicious)

    with tab_sample:
        # Pick from pre-loaded real files, for a live demo without needing
        # to source an executable during a presentation.
        st.write(
            "Real files from data/dataset_test.csv (already header-extracted, "
            "same schema as training data), useful for a live demo without "
            "needing to source an executable yourself. This file ships with "
            "the assignment dataset and has **no ground-truth label** "
            "attached, the names alone (e.g. `Skype-8.10.0.9.exe` vs. a "
            "SHA-256 hash) are a hint, not a confirmed answer, treat the "
            "model's verdict here as a demo output to discuss, not a "
            "graded prediction."
        )
        demo = load_demo_files()
        if demo is None:
            st.info("No data/dataset_test.csv found.")
        else:
            choice = st.selectbox("Pick a file", options=list(range(len(demo))),
                                   format_func=lambda i: demo["Name"].iloc[i])
            row_df = demo.loc[[choice], feature_order]
            render_verdict(pipeline, explainer, feature_order, row_df)

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
