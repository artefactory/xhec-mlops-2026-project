from pathlib import Path

TARGET_COL = "rings"

# parents[2] goes up from src/modelling/config.py to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_OBJECTS_DIRPATH = PROJECT_ROOT / "src" / "web_service" / "local_objects"
MODEL_PATH = LOCAL_OBJECTS_DIRPATH / "model.pkl"
DV_PATH = LOCAL_OBJECTS_DIRPATH / "dict_vectorizer.pkl"
TRAINSET_PATH = PROJECT_ROOT / "data" / "abalone.csv"
