import os

# Path to the root directory which contains the src directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to inputs
INPUT_DIR = os.path.join(ROOT_DIR, "inputs")
# File path for input schema file
INPUT_SCHEMA_DIR = os.path.join(INPUT_DIR, "schema")
# Path to data directory inside inputs directory
DATA_DIR = os.path.join(INPUT_DIR, "data")
# Path to model directory
DB_DIR_PATH = os.path.join(ROOT_DIR, "db")
# Path to the database file
DB_FILE_PATH = os.path.join(DB_DIR_PATH, "db.joblib")
# Path to src directory
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
