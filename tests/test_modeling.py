import pandas as pd
import pytest

from modules.modeling import train_model, DatasetValidationError
from services.session import get_session_dir

@pytest.mark.parametrize("model_type", ["Linear Regression", "Elastic Net Regression", "Gradient Boosting Regression"])
@pytest.mark.parametrize("target", ["age", "score"])
def test_train_model_regr(full_df, model_type, target):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    results, input_info = train_model(full_df, target, model_type, session_id)
    assert "R2" in results
    assert len(input_info) == len(full_df.columns) - 1

def test_train_error(full_df):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    with pytest.raises(ValueError):
        train_model(full_df, target="age", model_type="Random Forest", session_id=session_id)


@pytest.mark.parametrize("model_type", ["Logistic Regression", "Random Forest", "Decision Tree"])
@pytest.mark.parametrize("target", ["city", "category"])
def test_train_model_classif(full_df, model_type, target):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    results, input_info = train_model(full_df, target, model_type, session_id)
    assert "Accuracy" in results
    assert len(input_info) == len(full_df.columns) - 1

def test_mixed_types_error(full_df, mixed_col):
    session_dir = get_session_dir()  
    session_id = session_dir.name

    df = pd.concat([full_df, mixed_col], axis=1)
    with pytest.raises(DatasetValidationError):
        train_model(df, target="city", model_type="Logistic Regression", session_id=session_id)