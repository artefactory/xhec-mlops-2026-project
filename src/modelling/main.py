# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse

from config import MODELS_DIRPATH, RANDOM_STATE, TEST_SIZE
from loguru import logger
from predicting import predict
from preprocessing import extract_x_y, load_data, preprocess_data
from sklearn.model_selection import train_test_split
from training import evaluate_model, train_model
from utils import pickle_object


def main(trainset_path: str) -> None:
    """Train a model using the data at the given path and save the model and the DictVectorizer (pickle)."""
    df = load_data(trainset_path)
    df = preprocess_data(df)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    x_train, y_train, dv = extract_x_y(train_df)
    x_test, y_test, _ = extract_x_y(test_df, dv=dv)

    model = train_model(x_train, y_train)
    rmse = evaluate_model(y_test, predict(x_test, model))
    logger.info(f"Test RMSE: {rmse:.3f}")

    pickle_object(model, MODELS_DIRPATH / "model.pkl")
    pickle_object(dv, MODELS_DIRPATH / "dv.pkl")
    logger.info(f"Model and DictVectorizer saved in {MODELS_DIRPATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
