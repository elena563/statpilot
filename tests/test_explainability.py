import pandas as pd

from modules.explainability import preprocess


def test_preprocess_bool_to_int():
    X = pd.DataFrame({"flag": [True, False, True]})
    out = preprocess(X)
    assert out["flag"].tolist() == [1, 0, 1]


def test_preprocess_categorical_dummies():
    X = pd.DataFrame({"city": ["Milano", "Roma", "Milano"]})
    out = preprocess(X)
    assert "city_Roma" in out.columns
    assert out["city_Roma"].tolist() == [0, 1, 0]


def test_preprocess_fillna_with_mean():
    X = pd.DataFrame({"num": [1.0, None, 3.0]})
    out = preprocess(X)
    assert out["num"].tolist() == [1.0, 2.0, 3.0]


def test_preprocess_returns_float():
    X = pd.DataFrame({"a": [1, 2], "b": [True, False]})
    out = preprocess(X)
    assert (out.dtypes == "float").all()
