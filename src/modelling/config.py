from pathlib import Path

# [2] means go up 2 levels from the current file -> from ./src/modelling/config.py to ./
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRPATH = PROJECT_ROOT / "data"
MODELS_DIRPATH = PROJECT_ROOT / "src" / "web_service" / "local_objects"

TARGET = "rings"
CATEGORICAL_COLS = ["sex"]
NUMERICAL_COLS = ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
