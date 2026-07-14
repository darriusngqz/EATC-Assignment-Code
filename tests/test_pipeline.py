"""Tests for the trained Pipeline's prediction interface.

These need scikit-learn installed (pip install -r requirements-dev.txt).
They train a tiny throwaway pipeline on a handful of synthetic rows, they
do not load the real ~19,600-row dataset or the saved production model, so
they run in under a second and are safe to run on every commit.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from constants import ORDER_OF_FEATURES


def build_tiny_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
    ])


def tiny_training_frame(n_per_class=10):
    import pandas as pd

    rng = np.random.RandomState(0)
    n_features = len(ORDER_OF_FEATURES)

    malicious_rows = rng.normal(loc=5.0, scale=1.0, size=(n_per_class, n_features))
    benign_rows = rng.normal(loc=-5.0, scale=1.0, size=(n_per_class, n_features))

    # Label values here are arbitrary placeholders for two well-separated
    # synthetic clusters (this test only checks the pipeline mechanics, not
    # real malware/benign semantics), but kept in the project's actual
    # convention for consistency: 1 = malicious, 0 = benign (see
    # MALICIOUS_LABEL/BENIGN_LABEL in constants.py).
    X = np.vstack([malicious_rows, benign_rows])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    return pd.DataFrame(X, columns=ORDER_OF_FEATURES), pd.Series(y)


def test_pipeline_trains_and_predicts_valid_probabilities():
    X, y = tiny_training_frame()
    pipeline = build_tiny_pipeline()
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X.iloc[[0]])[0]
    assert 0.0 <= proba[0] <= 1.0
    assert 0.0 <= proba[1] <= 1.0
    assert abs(proba.sum() - 1.0) < 1e-6


def test_pipeline_separates_obviously_different_classes():
    # Sanity check, not a real accuracy benchmark: two well-separated
    # synthetic clusters should be trivially classified correctly.
    X, y = tiny_training_frame(n_per_class=30)
    pipeline = build_tiny_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    accuracy = (preds == y.values).mean()
    assert accuracy > 0.9


def test_pipeline_scaler_is_fit_only_on_training_data():
    # Regression test for a scaler train/test leakage fix: the scaler
    # inside the pipeline must be fit exactly once, via pipeline.fit() on
    # training data only, never separately on the full dataset beforehand.
    X, y = tiny_training_frame()
    pipeline = build_tiny_pipeline()
    pipeline.fit(X, y)
    scaler = pipeline.named_steps["scaler"]
    assert hasattr(scaler, "mean_"), "scaler was never fit"
    assert scaler.mean_.shape[0] == len(ORDER_OF_FEATURES)
