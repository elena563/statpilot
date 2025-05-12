import os
import shutil
from flask import Flask, flash, redirect, render_template, request, send_from_directory
import time
import threading
from modules.analysis import analyze_csv

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

@app.route("/model")
def model():

    model = request.form.get('model')

    if request.method == 'POST':
        file = request.files['dataset']
        if not file:
            return render_template("analysis.html", error='No file submitted')
        
        return render_template("analysis.html")
    else:
        return render_template("modeling.html")

@app.route("/explain")
def explain():
    return render_template("explainability.html")

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(directory=os.path.dirname(filename),
                               filename=os.path.basename(filename),
                               as_attachment=True)

if __name__ == '__main__':
    with app.app_context():
        init_cleanup()
    app.run(debug=True)