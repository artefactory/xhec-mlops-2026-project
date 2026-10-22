# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from config import DV_PATH, MODEL_PATH
from predicting import predict
from preprocessing import clean_data, extract_x_y, load_data
from sklearn.model_selection import train_test_split
from training import evaluate_model, train_model
from utils import pickle_object


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle).

    Args:
        trainset_path: Path of the abalone CSV file.
    """
    # Read and clean data
    df = clean_data(load_data(trainset_path))
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Preprocess data (the DictVectorizer is fitted on the train set only)
    x_train, y_train, dv = extract_x_y(df_train)
    x_test, y_test, _ = extract_x_y(df_test, dv=dv)

    # Train and evaluate model
    model = train_model(x_train, y_train)
    rmse = evaluate_model(y_test, predict(x_test, model))
    print(f"Test RMSE: {rmse:.3f}")

    # Pickle the DictVectorizer and the model
    pickle_object(dv, DV_PATH)
    pickle_object(model, MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=Path, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
