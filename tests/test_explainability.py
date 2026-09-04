import json

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression

from modules.explainability import compute_shap_values, preprocess


def _make_onnx_model(n_features):
    X = np.random.RandomState(0).rand(100, n_features).astype("float32")
    y = np.random.RandomState(0).rand(100).astype("float32")
    model = LinearRegression().fit(X, y)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = to_onnx(model, initial_types=initial_type)
    return ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])


def _make_session_with_columns(tmp_path, columns):
    session_id = "test-session"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    with open(session_dir / "columns.json", "w") as f:
        json.dump(columns, f)
    return session_id, tmp_path


def test_preprocess_bool_to_int():
    X = pd.DataFrame({"flag": [True, False, True]})
    out = preprocess(X)
    assert out["flag"].tolist() == [1, 0, 1]


def test_preprocess_categorical_dummies():
    X = pd.DataFrame({"city": ["Milano", "Roma", "Milano"]})
    out = preprocess(X)
    assert "city_Milano" in out.columns
    assert "city_Roma" in out.columns
    assert out["city_Milano"].tolist() == [1.0, 0.0, 1.0]
    assert out["city_Roma"].tolist() == [0.0, 1.0, 0.0]


def test_preprocess_fillna_with_mean():
    X = pd.DataFrame({"num": [1.0, None, 3.0]})
    out = preprocess(X)
    assert out["num"].tolist() == [1.0, 2.0, 3.0]


def test_preprocess_returns_float():
    X = pd.DataFrame({"a": [1, 2], "b": [True, False]})
    out = preprocess(X)
    assert (out.dtypes == "float32").all()


def test_compute_shap_small_dataset_no_sampling(tmp_path):
    onnx_model = _make_onnx_model(3)
    X = pd.DataFrame(np.random.RandomState(0).rand(30, 3).astype("float32"), columns=["a", "b", "c"])
    session_id, temp_dir = _make_session_with_columns(tmp_path, ["a", "b", "c"])
    # Monkey-patch session_path to use temp_dir
    import modules.explainability as expl_mod

    original_session_path = expl_mod.session_path
    expl_mod.session_path = lambda sid, fname: temp_dir / fname
    try:
        shap_values, X_sub, explainer, target_pos = compute_shap_values(onnx_model, X, session_id, target_idx=5)
        assert X_sub.shape[0] == 30
        assert target_pos == 5
    finally:
        expl_mod.session_path = original_session_path


def test_compute_shap_large_dataset_sampling(tmp_path, numeric_df_54):
    onnx_model = _make_onnx_model(3)
    session_id, temp_dir = _make_session_with_columns(tmp_path, ["a", "b", "c"])
    import modules.explainability as expl_mod

    original_session_path = expl_mod.session_path
    expl_mod.session_path = lambda sid, fname: temp_dir / fname
    try:
        shap_values, X_sub, explainer, target_pos = compute_shap_values(onnx_model, numeric_df_54, session_id, target_idx=40)
        assert X_sub.shape[0] <= 200
        assert target_pos is not None
        assert 0 <= target_pos < X_sub.shape[0]
        np.testing.assert_array_equal(X_sub.iloc[target_pos].values, numeric_df_54.iloc[40].values)
    finally:
        expl_mod.session_path = original_session_path


def test_compute_shap_large_dataset_no_target(tmp_path, numeric_df_54):
    onnx_model = _make_onnx_model(3)
    session_id, temp_dir = _make_session_with_columns(tmp_path, ["a", "b", "c"])
    import modules.explainability as expl_mod

    original_session_path = expl_mod.session_path
    expl_mod.session_path = lambda sid, fname: temp_dir / fname
    try:
        shap_values, X_sub, explainer, target_pos = compute_shap_values(onnx_model, numeric_df_54, session_id)
        assert X_sub.shape[0] <= 200
        assert target_pos is None
    finally:
        expl_mod.session_path = original_session_path
