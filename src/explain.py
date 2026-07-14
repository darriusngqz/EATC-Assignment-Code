"""SHAP-based explainability for the deployed classical Pipeline. SHAP
explains a prediction after the model already made it, like a tug-of-war:
each feature value pulls the verdict toward malicious or toward benign.

Label convention: Malware == 1 malicious, 0 benign (see constants.py).
shap_values[-1] is the SHAP array for class 1 (malicious): positive pushes
toward malicious, negative pushes toward benign.
"""
import shap


# Sets up a SHAP explainer for one fitted pipeline. Only the "model" step
# is passed to SHAP, TreeExplainer reads the tree structure directly and
# has no use for the scaler.
def build_explainer(pipeline):
    model = pipeline.named_steps["model"]
    return shap.TreeExplainer(model)


# Explains one prediction: which features pushed it toward malicious, and
# which pushed it toward benign. raw_row is unscaled, shape (1, n_features).
# n is capped at half the feature count, so with a small feature set the
# top-n from each end can't overlap and double-count the same feature.
def explain_prediction(explainer, pipeline, raw_row, feature_names, n=8):
    # Scale the same way training data was scaled, the model expects that.
    scaler = pipeline.named_steps["scaler"]
    scaled_row = scaler.transform(raw_row)
    shap_values = explainer.shap_values(scaled_row)

    # Some SHAP/model versions return a list of per-class arrays instead of
    # one array; normalise here. Last array = class 1 = malicious.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    values = shap_values[0]  # single file, so take that one row

    n = max(1, min(n, len(feature_names) // 2))
    # Sort smallest (most benign-pushing) to largest (most malicious-pushing).
    ranked = sorted(zip(feature_names, values), key=lambda t: t[1])
    toward_benign = ranked[:n]
    toward_malicious = ranked[::-1][:n]  # reversed, so most positive first
    return toward_malicious, toward_benign


# Run once, offline, to sanity-check the model isn't relying on a
# shortcut feature (e.g. TimeDateStamp, which reflects when a sample was
# compiled, not malware behaviour, and shouldn't dominate).
def global_feature_importance(explainer, X_sample, feature_names, n=15):
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    # Average absolute pull per feature across all rows, direction doesn't
    # matter here, only how hard a feature pulls overall.
    mean_abs = abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda t: t[1], reverse=True)
    return ranked[:n]
