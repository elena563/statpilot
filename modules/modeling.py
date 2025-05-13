import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def train_model(df, y, model_type):

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

    X = df.drop('target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if task == 'regr':
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
    else:
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

    
    joblib.dump(model, 'static/model.pkl')


    return []