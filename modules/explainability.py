import shap
import matplotlib.pyplot as plt
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

    # nan filling
    X = X.fillna(X.mean(numeric_only=True))
    return X.astype(float)

def setup_explainer(model, X_test):
    # check for model type to choose the right explainer
    model_type = type(model).__name__
    X_test_prep = preprocess(X_test)

    if model_type in ['LogisticRegression', 'LinearRegression', 'ElasticNet']:
        explainer = shap.LinearExplainer(model, X_test_prep)
    elif model_type in ['RandomForestClassifier', 'GradientBoostingRegressor', 'DecisionTreeClassifier']:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_prep)

    return shap_values, X_test_prep, explainer

def explain_global(model, X_test):
    shap_values, X_test_prep, _ = setup_explainer(model, X_test)

    shap.summary_plot(shap_values, X_test_prep)

    session_dir = get_session_dir()
    path = str(Path(session_dir) / "distributions.png").replace('\\', '/')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    return path

def explain_local(obs_index, model, X_test):
    shap_values, X_test_prep, explainer = setup_explainer(model, X_test)

    plots = []
    session_dir = get_session_dir()

    shap.force_plot(
    shap_values[obs_index].base_values,
    shap_values[obs_index].values,
    X_test_prep.iloc[obs_index],
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