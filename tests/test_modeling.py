import pandas as pd
import pytest

from modules.modeling import train_model, DatasetValidationError
from services.session import get_session_dir

@pytest.mark.parametrize("model_type", ["Linear Regression", "Elastic Net Regression", "Gradient Boosting Regression"])
@pytest.mark.parametrize("target", ["age", "score"])
def test_train_model_regr(base_df, model_type, target):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    results, input_info, _ = train_model(base_df, target, model_type, session_id)
    assert "R2" in results
    assert len(input_info) == len(base_df.columns) - 1

def test_train_error(base_df):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    with pytest.raises(ValueError):
        train_model(base_df, target="age", model_type="Random Forest", session_id=session_id)


@pytest.mark.parametrize("model_type", ["Logistic Regression", "Random Forest", "Decision Tree"])
@pytest.mark.parametrize("target", ["city", "category"])
def test_train_model_classif(base_df, model_type, target):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    results, input_info, _ = train_model(base_df, target, model_type, session_id)
    assert "Accuracy" in results
    assert len(input_info) == len(base_df.columns) - 1

def test_mixed_types_error(base_df, mixed_col):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    df = pd.concat([base_df, mixed_col], axis=1)
    with pytest.raises(DatasetValidationError):
        train_model(df, target="city", model_type="Logistic Regression", session_id=session_id)

def test_nan_feature_drop(base_df, nan_col):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    df = pd.concat([base_df, nan_col], axis=1)
    _, input_info, _ = train_model(df, target="city", model_type="Logistic Regression", session_id=session_id)
    assert len(input_info) == len(df.columns) - 1
    assert "nan col" not in input_info

def test_nan_target_error(base_df, nan_col):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    df = pd.concat([base_df, nan_col], axis=1)
    with pytest.raises(DatasetValidationError):
        train_model(df, target="nan col", model_type="Logistic Regression", session_id=session_id)

def test_const_feature_drop(base_df, const_col):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    df = pd.concat([base_df, const_col], axis=1)
    _, input_info, _ = train_model(df, target="age", model_type="Linear Regression", session_id=session_id)
    assert len(input_info) == len(df.columns) - 1
    assert "nan col" not in input_info

def test_const_target_error(base_df, const_col):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    df = pd.concat([base_df, const_col], axis=1)
    with pytest.raises(DatasetValidationError):
        train_model(df, target="const col", model_type="Linear Regression", session_id=session_id)

def test_unbalanced_target_warning(unbalanced_df):
    session_dir = get_session_dir()  
    session_id = session_dir.name
    
    _, _, warnings = train_model(unbalanced_df, target="pet", model_type="Logistic Regression", session_id=session_id)
    assert len(warnings) == 1

def test_one_sample_col_error(base_df, one_sample_col):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    df = pd.concat([base_df, one_sample_col], axis=1)
    with pytest.raises(DatasetValidationError):
        train_model(df, target="one sample col", model_type="Logistic Regression", session_id=session_id)

def test_small_df_error(base_df):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    df = base_df.head(9)
    with pytest.raises(DatasetValidationError):
        train_model(df, target="city", model_type="Logistic Regression", session_id=session_id)