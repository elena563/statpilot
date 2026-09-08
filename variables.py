import os

TEST_SIZE = 0.2
RANDOM_STATE = 42
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMP_DIR = os.environ.get("STATPILOT_TEMP_DIR", "/temp")
CLEANUP_MAX_AGE_HOURS = 24
CLEANUP_INTERVAL_HOURS = 12
