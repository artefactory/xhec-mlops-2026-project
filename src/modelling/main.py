# This module is the training entrypoint: it runs the Prefect training flow on the given data.

import argparse

from workflows import train_model_workflow


def main(trainset_path: str) -> None:
    """Train a model using the data at the given path and save the model and the DictVectorizer (pickle)."""
    train_model_workflow(trainset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
