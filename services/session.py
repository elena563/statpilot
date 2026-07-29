import os
from pathlib import Path
import re
import uuid

from variables import TEMP_DIR

UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
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