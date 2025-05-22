from flask import Flask, render_template
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

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
        case 'Decision Tree':
            model = DecisionTreeClassifier() 
            task = 'classif'
    if task != target_type:
        raise ValueError(f"Modello '{model_type}' non supportato per target {target}")


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

    path = Path("static") / "temp" / session_id / "xtest.csv"
    X_test.to_csv(path, index=False)

    dfx = df.drop(columns=[target])
    input_info = []
    for col in dfx.columns:
        dtype = dfx[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            input_info.append({"name": col, "type": 'number'})
        elif pd.api.types.is_bool_dtype(dtype):
            input_info.append({"name": col, "type": 'bool'})
        else:
            categories = dfx[col].dropna().unique().tolist()
            input_info.append({'name': col, 'type': 'text', 'choices': categories})

    return results, input_info

def test_model(dfx, model, input_data):

    input_series = pd.Series(input_data)
    X = input_series.to_frame().T

    X = X.astype(dfx.dtypes.to_dict())
    X = pd.get_dummies(X)
    y_pred = model.predict(X)
    feature_names = list(dfx.columns)
    row = X.iloc[0].values.reshape(1, -1)
    row_list = row.flatten().tolist()
    y_pred2 = y_pred[0]

    return y_pred2, row_list, feature_names