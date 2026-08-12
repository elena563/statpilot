import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from services.session import session_path


def preprocess(X):
    # boolean values encoding
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    # categorical values encoding
    cat_cols = X.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.fillna(X.mean(numeric_only=True))
    return X.astype(float)


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


def setup_explainer(model, X_test):
    # use KernelExplainer for ONNX models
    X_test_prep = preprocess(X_test)

    input_name = model.get_inputs()[0].name
    background = shap.sample(X_test_prep, 50)
    if len(model.get_outputs()) > 1:
        explainer = shap.KernelExplainer(lambda X: _onnx_predict_proba(model, X, input_name), background)
    else:
        explainer = shap.KernelExplainer(lambda X: _onnx_predict(model, X, input_name), background)

    return X_test_prep, explainer


def compute_shap_values(model, X):
    X_test_prep, explainer = setup_explainer(model, X)

    n = X_test_prep.shape[0]
    if n < 50:
        X_sub = X_test_prep
    else:
        X_sub = shap.sample(X_test_prep, min(200, int(n * 0.3)))
    shap_values = explainer.shap_values(X_sub, nsamples=int(np.clip(X_test_prep.shape[1] * 20, 500, 2000)))

    if isinstance(shap_values, list):  # multi-class classification
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_sub, explainer


def explain_global(model, X_test, session_id):
    shap_values, X_sub, _ = compute_shap_values(model, X_test)

    shap.summary_plot(shap_values, X_sub, show=False)

    path = str(session_path(session_id, "var-importance.png")).replace("\\", "/")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return "var-importance.png"


def explain_local(obs_index, model, X_test, session_id):
    shap_values, X_sub, explainer = compute_shap_values(model, X_test)

    plots = []
    session_dir = session_path(session_id, "")

    if isinstance(shap_values, list):
        c_idx = 0
        base_val = explainer.expected_value[c_idx]
        vals = shap_values[c_idx][obs_index]
    else:
        base_val = explainer.expected_value
        vals = shap_values[obs_index]
        if isinstance(base_val, list | np.ndarray):
            base_val = base_val[0]

    shap.force_plot(base_val, vals, X_sub.iloc[obs_index], matplotlib=True)

    exp = shap.Explanation(values=shap_values[obs_index], base_values=base_val, data=X_sub.iloc[obs_index].values, feature_names=X_sub.columns.tolist())

    plt.savefig(session_dir / "forceplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    plots.append("forceplot.png")

    shap.plots.waterfall(exp)
    plt.savefig(session_dir / "waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()
    plots.append("waterfall.png")

    return plots
