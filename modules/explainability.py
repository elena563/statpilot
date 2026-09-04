import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from services.session import session_path


def preprocess(X):
    # boolean values encoding
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    # categorical values encoding — must match training: pd.get_dummies(X)
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)

    X = X.fillna(X.mean(numeric_only=True))
    return X.astype("float32")


def _align_columns(X, session_id):
    """Align columns with training columns from columns.json."""
    columns_path = session_path(session_id, "columns.json")
    if not columns_path.exists():
        return X
    with open(columns_path) as f:
        expected_columns = json.load(f)
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0
    return X[expected_columns]


def _onnx_predict(model, X, input_name):
    return model.run(None, {input_name: np.asarray(X, dtype=np.float32)})[0].flatten()


def _onnx_predict_proba(model, X, input_name):
    outputs = model.run(None, {input_name: np.asarray(X, dtype=np.float32)})
    if len(outputs) > 1:
        out = outputs[-1]
        if isinstance(out, list):
            keys = list(out[0].keys())
            return np.array([[r.get(k, 0.0) for k in keys] for r in out], dtype=np.float32)
        return out
    return outputs[0]


def setup_explainer(model, X_test, session_id):
    # use KernelExplainer for ONNX models
    X_test_prep = preprocess(X_test)
    X_test_prep = _align_columns(X_test_prep, session_id)

    input_name = model.get_inputs()[0].name
    background = shap.sample(X_test_prep, 50)
    if len(model.get_outputs()) > 1:
        explainer = shap.KernelExplainer(lambda X: _onnx_predict_proba(model, X, input_name), background)
    else:
        explainer = shap.KernelExplainer(lambda X: _onnx_predict(model, X, input_name), background)

    return X_test_prep, explainer


def compute_shap_values(model, X, session_id, target_idx=None):
    X_test_prep, explainer = setup_explainer(model, X, session_id)

    n = X_test_prep.shape[0]
    if n < 50:
        X_sub = X_test_prep
        target_pos = target_idx if target_idx is not None else None
    else:
        X_sub = shap.sample(X_test_prep, min(200, int(n * 0.3)))
        if target_idx is not None:
            target_row = X_test_prep.iloc[[target_idx]]
            if not X_sub.index.isin(target_row.index).any():
                X_sub = pd.concat([X_sub, target_row])
            target_pos = X_sub.index.get_loc(target_row.index[0])
        else:
            target_pos = None

    shap_values = explainer.shap_values(X_sub, nsamples=int(np.clip(X_test_prep.shape[1] * 20, 500, 2000)))

    if isinstance(shap_values, list):  # multi-class classification
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_sub, explainer, target_pos


def explain_global(model, X_test, session_id):
    shap_values, X_sub, _, _ = compute_shap_values(model, X_test, session_id)

    shap.summary_plot(shap_values, X_sub, show=False)

    path = str(session_path(session_id, "var-importance.png")).replace("\\", "/")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return "var-importance.png"


def explain_local(obs_index, model, X_test, session_id):
    shap_values, X_sub, explainer, target_pos = compute_shap_values(model, X_test, session_id, target_idx=obs_index)

    plots = []

    if isinstance(shap_values, list):
        c_idx = 0
        base_val = explainer.expected_value[c_idx]
        vals = shap_values[c_idx][target_pos]
    else:
        base_val = explainer.expected_value
        vals = shap_values[target_pos]
        if isinstance(base_val, list | np.ndarray):
            base_val = base_val[0]

    shap.force_plot(base_val, vals, X_sub.iloc[target_pos], matplotlib=True)

    exp = shap.Explanation(values=shap_values[target_pos], base_values=base_val, data=X_sub.iloc[target_pos].values, feature_names=X_sub.columns.tolist())

    plt.savefig(session_path(session_id, "forceplot.png"), dpi=300, bbox_inches="tight")
    plt.close()
    plots.append("forceplot.png")

    shap.plots.waterfall(exp)
    plt.savefig(session_path(session_id, "waterfall.png"), dpi=300, bbox_inches="tight")
    plt.close()
    plots.append("waterfall.png")

    return plots
