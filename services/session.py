import joblib
import os
from pathlib import Path
import re
import shutil
import time
import threading
import uuid
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from variables import TEMP_DIR, CLEANUP_MAX_AGE_HOURS, CLEANUP_INTERVAL_HOURS

UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)

ALLOWED_MODELS = (
        RandomForestClassifier, RandomForestRegressor,
        LogisticRegression, LinearRegression,
        DecisionTreeClassifier, DecisionTreeRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
    )

# session logic
def get_session_dir() -> Path:
    session_id = str(uuid.uuid4())
    session_dir = Path(TEMP_DIR) / session_id
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def validate_session_id(session_id: str | None) -> str:
    """Returns the session ID if valid, raises ValueError otherwise."""
    if not session_id or not UUID_RE.match(session_id):
        raise ValueError("Invalid or missing session ID")
    return session_id

def session_path(session_id: str, filename: str) -> Path:
    return Path(TEMP_DIR) / session_id / filename

# loading logic
def load_dataframe(session_id: str, filename: str = "dataset.csv") -> pd.DataFrame:
    path = session_path(session_id, filename)
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        raise ValueError("The CSV file is malformed, please check the file and try again.")
    except UnicodeDecodeError:
        raise ValueError("The CSV file has an unsupported encoding. Please save it in UTF-8.")
    except FileNotFoundError:
        raise FileNotFoundError("The session has expired. Please upload the file again.")

def load_model(session_id: str, validate: bool = False):
    path = session_path(session_id, "model.pkl")
    try:
        model = joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("Model not found, session expired")
    except Exception:
        raise ValueError("Error loading the model")
    
    if validate and not isinstance(model, ALLOWED_MODELS):
        raise ValueError("Not a valid model type. Please upload a supported model.")
    
    return model


# clean temporary directories
def cleanup_temp_dirs(max_age_hours=CLEANUP_MAX_AGE_HOURS):
    temp_dir = os.path.join(TEMP_DIR)
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

def _cleanup_loop(interval_hours=CLEANUP_INTERVAL_HOURS):
    while True:
        cleanup_temp_dirs()
        time.sleep(interval_hours * 3600)

def init_cleanup():
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    cleanup_temp_dirs()
    
    t = threading.Thread(target=_cleanup_loop)
    t.start()