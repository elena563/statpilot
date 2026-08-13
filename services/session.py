import hashlib
import hmac
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import onnxruntime as ort
import pandas as pd

from variables import CLEANUP_INTERVAL_HOURS, CLEANUP_MAX_AGE_HOURS, TEMP_DIR

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


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


def make_session_token(session_id: str) -> str:
    secret = os.environ["SECRET_KEY"].encode()
    return hmac.new(secret, session_id.encode(), hashlib.sha256).hexdigest()


def verify_session_token(session_id: str, token: str | None) -> bool:
    if not token:
        return False
    expected = make_session_token(session_id)
    return hmac.compare_digest(expected, token)


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


def load_model(session_id: str):
    path = session_path(session_id, "model.onnx")
    if not path.exists():
        raise FileNotFoundError("Model not found, session expired")
    try:
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception:
        raise ValueError("Error loading the model")


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
    t.daemon = True
    t.start()
