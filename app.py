import os
import shutil
from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
from pathlib import Path
import time
import threading
from modules.analysis import analyze_csv, get_session_dir, read_csv_sep
from modules.modeling import train_model, test_model
from modules.explainability import explain_global, explain_local

# configure application
app = Flask(__name__)
app.debug = True

# prevent caching
@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# clean temporary directories
def cleanup_temp_dirs(max_age_hours=24):
    temp_dir = os.path.join("static", "temp")
    if not os.path.exists(temp_dir):
        return
        
    current_time = time.time()
    for session_dir in os.listdir(temp_dir):
        dir_path = os.path.join(temp_dir, session_dir)
        if os.path.isdir(dir_path):
            dir_age = current_time - os.path.getmtime(dir_path)
            if dir_age > (max_age_hours * 3600): 
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed: {dir_path}")
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")

def schedule_cleanup(interval_hours=12):
    while True:
        cleanup_temp_dirs()
        time.sleep(interval_hours * 3600)

def init_cleanup():
    os.makedirs(os.path.join("static", "temp"), exist_ok=True)
    
    # clean at start
    cleanup_temp_dirs()
    
    # periodic cleaning
    cleanup_thread = threading.Thread(target=schedule_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == 'POST':
        file = request.files['dataset']
        if not file:
            return render_template("analysis.html", error='No file submitted')
        
        # file extension check
        filename = file.filename
        _, file_ext = os.path.splitext(filename)
        if file_ext.lower() != '.csv':
            return render_template("explainability.html", error='dataset file must be a CSV (.csv)')

        results = analyze_csv(file)
        return render_template("analysis.html", results=results)
    else:
        return render_template("analysis.html")


@app.route("/model", methods=["GET", "POST"])
def model():

    session_dir = get_session_dir()
    path = session_dir / "dataset.csv"
    form_type = request.form.get("form_type")

    if request.method == 'POST':

        if form_type == 'train_form':

            if 'dataset' in request.files:
                file = request.files['dataset']
                if not file:
                    return render_template("modeling.html", error='No file submitted')
                
                # file extension check
                filename = file.filename
                _, file_ext = os.path.splitext(filename)
                if file_ext.lower() != '.csv':
                    return render_template("explainability.html", error='dataset file must be a CSV (.csv)')

                # temporary save dataset to pass it to the form
                session_dir = get_session_dir()  

                session_id = session_dir.name

                path = session_dir / "dataset.csv"
                df = read_csv_sep(file) 
                df.to_csv(path, index=False)
                
                columns = df.columns.to_list()
                return render_template("modeling.html", columns=columns, session_id=session_id)
            
            elif 'target' in request.form:
                session_id = request.form.get('session_id')  # prendi l'id inviato
                session_dir = Path("static") / "temp" / session_id
                path = session_dir / "dataset.csv"
                df = pd.read_csv(path)
                model_path = session_dir / "model.pkl"

                model_type = request.form.get('model')
                target = request.form.get('target')
                (session_dir / "target.txt").write_text(target)
                try:
                    results, input_info = train_model(df, target, model_type, session_id)
                except ValueError as e:
                    return render_template("modeling.html", error=str(e), session_id=session_id)

            return render_template("modeling.html", results=results, session_id=session_id, input_info=input_info)
        
        elif form_type == 'test_form':
            session_id = request.form.get("session_id")
            session_dir = Path("static") / "temp" / session_id

            path = session_dir / "dataset.csv"
            model_path = session_dir / "model.pkl"

            df = pd.read_csv(path)
            model = joblib.load(model_path)

            target_path = session_dir / "target.txt"
            target = target_path.read_text().strip()
            dfx = df.drop(columns=[target])

            input_data = {col: request.form.get(col) for col in dfx.columns}
            if None in input_data.values():
                return render_template("modeling.html", error="Please insert a value for each input feature")

            result, row_list, feature_names = test_model(dfx, model, input_data, session_id)
            return render_template("modeling.html", result=result, row_list=row_list, feature_names=feature_names, session_id=session_id, target=target)
    else:
        return render_template("modeling.html")


@app.route("/explain", methods=["GET", "POST"])
def explain():

    if request.method == 'POST':
        form_type = request.form.get("form_type")
        
        if form_type == 'global_form':
            xtest_file = request.files['xtest']
            model_file = request.files['model']
            if not xtest_file or not model_file:
                return render_template("explainability.html", error='No file submitted')
            
            # files extensions check
            xtest_filename = xtest_file.filename
            _, xtest_ext = os.path.splitext(xtest_filename)
            if xtest_ext.lower() != '.csv':
                return render_template("explainability.html", error='xtest file must be a CSV (.csv)')
            model_filename = model_file.filename
            _, model_ext = os.path.splitext(model_filename)
            if model_ext.lower() != '.pkl':
                return render_template("explainability.html", error='model file must be a Pickle (.pkl)')

            # temporary save dataset and model to pass them to the form
            session_dir = get_session_dir()  
            session_id = session_dir.name

            X_path = session_dir / "xtest.csv"
            model_path = session_dir / "model.pkl"
    
            xtest_file.save(X_path)
            model_file.save(model_path)

            X_test = pd.read_csv(X_path)
            model = joblib.load(model_file.stream) 

            summary_plot = explain_global(model, X_test)
            return render_template("explainability.html", summary_plot=summary_plot, session_id=session_id)
        
        elif form_type == 'local_form':
            session_id = request.form.get("session_id")
            X_test = pd.read_csv(f"static/temp/{session_id}/xtest.csv")
            model = joblib.load(f"static/temp/{session_id}/model.pkl")

            obs_index = int(request.form.get("obs"))
            row = X_test.iloc[obs_index].values.reshape(1, -1)
            y_pred = model.predict(row)
            feature_names = list(X_test.columns)
            row_list = row.flatten().tolist()
            y_pred2 = y_pred[0]

            plots = explain_local(obs_index, model, X_test)

            return render_template("explainability.html", plots=plots, row_list=row_list, y_pred2=y_pred2, feature_names=feature_names)
    else:
        return render_template("explainability.html")


@app.route("/download")
def download():
    session_id = request.args.get('session_id')
    file = request.args.get('file') 
    if not session_id:
        return "Missing session ID", 400
    
    if file == 'model':
        filename = "model.pkl"
    elif file == 'xtest':
        filename = "xtest.csv"

    path = Path("static") / "temp" / session_id / filename
    if not path.exists():
        return "File not found", 404

    return send_file(path, as_attachment=True, download_name=filename)

@app.route("/learn")
def learn():
    return render_template("learn.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
