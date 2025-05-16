import os
import shutil
from flask import Flask, render_template, request, send_from_directory, send_file
import pandas as pd
from pathlib import Path
import time
import threading
from modules.analysis import analyze_csv, get_session_dir
from modules.modeling import train_model

# configure application
app = Flask(__name__)
app.debug = True
app.secret_key = "sT4tz42!mylittlesecret"   # dev server key

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
    # Crea la directory temp se non esiste
    os.makedirs(os.path.join("static", "temp"), exist_ok=True)
    
    # Esegui pulizia all'avvio
    cleanup_temp_dirs()
    
    # Avvia thread per pulizia periodica
    cleanup_thread = threading.Thread(target=schedule_cleanup)
    cleanup_thread.daemon = True  # Il thread terminerà quando l'app si chiude
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
        results = analyze_csv(file)
        return render_template("analysis.html", results=results)
    else:
        return render_template("analysis.html")

@app.route("/model", methods=["GET", "POST"])
def model():

    session_dir = get_session_dir()
    path = session_dir / "dataset.csv"

    if request.method == 'POST':

        if 'dataset' in request.files:
            file = request.files['dataset']
            if not file:
                return render_template("modeling.html", error='No file submitted')
            session_dir = get_session_dir()  
        
            session_id = session_dir.name

            path = session_dir / "dataset.csv"
            df = pd.read_csv(file) 
            df.to_csv(path, index=False)
            
            columns = df.columns.to_list()
            return render_template("modeling.html", columns=columns, session_id=session_id)
        
        elif 'target' in request.form:
            session_id = request.form.get('session_id')  # prendi l'id inviato
            session_dir = Path("static") / "temp" / session_id
            path = session_dir / "dataset.csv"
            df = pd.read_csv(path)
            model_type = request.form.get('model')
            target = request.form.get('target')
            results = train_model(df, target, model_type, session_id=session_id)

        return render_template("modeling.html", results=results, session_id=session_id)
    else:
        return render_template("modeling.html")

@app.route("/explain")
def explain():
    return render_template("explainability.html")

@app.route("/download_model")
def download_model():
    session_id = request.args.get('session_id')
    if not session_id:
        return "Session ID mancante", 400

    path = Path("static") / "temp" / session_id / "model.pkl"
    if not path.exists():
        return "Modello non trovato", 404

    return send_file(path, as_attachment=True, download_name="model.pkl")

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(directory=os.path.dirname(filename),
                               filename=os.path.basename(filename),
                               as_attachment=True)

if __name__ == '__main__':
    with app.app_context():
        init_cleanup()
    app.run(debug=True)