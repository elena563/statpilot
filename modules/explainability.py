import shap
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path
from modules.analysis import get_session_dir

def preprocess(X):
    # boolean values encoding
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)

    # categorical values encoding
    cat_cols = X.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.fillna(X.mean(numeric_only=True))
    return X.astype(float)

def _onnx_predict(model, X, input_name):
    return model.run(None, {input_name: np.asarray(X, dtype=np.float32)})[0]

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
        explainer = shap.KernelExplainer(
            lambda X: _onnx_predict_proba(model, X, input_name), background)
    else:
        explainer = shap.KernelExplainer(
            lambda X: _onnx_predict(model, X, input_name), background)

    return X_test_prep, explainer

def compute_shap_values(model, X):
    X_test_prep, explainer = setup_explainer(model, X)
    X_sub = shap.sample(X_test_prep, min(200, int(X_test_prep.shape[0] * 0.3)))
    shap_values = explainer.shap_values(X_sub, nsamples=int(np.clip(X_test_prep.shape[1] * 20, 500, 2000)))

    if isinstance(shap_values, list):   # multi-class classification
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_sub

def explain_global(model, X_test):
    shap_values, X_sub = compute_shap_values(model, X_test)

    shap.summary_plot(shap_values, X_sub, show=False)

    session_dir = get_session_dir()
    path = str(Path(session_dir) / "distributions.png").replace('\\', '/')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    return path

def explain_local(obs_index, model, X_test):
    shap_values, X_sub = compute_shap_values(model, X_test)

    plots = []
    session_dir = get_session_dir()

    shap.force_plot(
    shap_values[obs_index].base_values,
    shap_values[obs_index].values,
    X_sub.iloc[obs_index],
    matplotlib=True
    )
    path = str(Path(session_dir) / "forceplot.png").replace('\\', '/')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(path)

    shap.plots.waterfall(shap_values[obs_index])
    path = str(Path(session_dir) / "waterfall.png").replace('\\', '/')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(path)

    return plots