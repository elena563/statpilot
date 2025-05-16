from flask import Flask, render_template
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from modules.analysis import get_session_dir

def train_model(df, target, model_type, session_id):

    X = df.drop(columns=[target])
    X = pd.get_dummies(X)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        target_type = 'classif'
    else:
        target_type = 'regr'
    task = None

    match model_type:
        case 'Linear Regression':
            model = LinearRegression()
            task = 'regr'
        case 'Elastic Net Regression':
            model = ElasticNet()
            task = 'regr'
        case 'Gradient Boosting Regression':
            model = GradientBoostingRegressor()
            task = 'regr'
        case 'Logistic Regression':
            model = LogisticRegression() 
            task = 'classif'
        case 'Random Forest':
            model = RandomForestClassifier()
            task = 'classif'
        case 'Naive Bayes':
            model = GaussianNB() 
            task = 'classif'
    try:
        if task != target_type:
            raise ValueError(f"Modello '{model_type}' non supportato per target {y}")
    except ValueError as e:
        return render_template("modeling.html", error=str(e))


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    results = {
            'Model type': model_type,
            'Target variable': target,
            'Feature names': ", ".join(X.columns)
    }

    if task == 'regr':
        results['R2'] = round(r2_score(y_test, y_pred), 3)
        results['MSE'] = round(mean_squared_error(y_test, y_pred), 3)
        results['RMSE'] = round(root_mean_squared_error(y_test, y_pred), 3)
    else:
        results['Accuracy'] = round(accuracy_score(y_test, y_pred), 3)
        results['Precision'] = round(precision_score(y_test, y_pred, average='weighted'), 3)
        results['Recall'] = round(recall_score(y_test, y_pred, average='weighted'), 3)
        results['F1'] = round(f1_score(y_test, y_pred, average='weighted'), 3)

    path = Path("static") / "temp" / session_id / "model.pkl"
    joblib.dump(model, path)

    return results