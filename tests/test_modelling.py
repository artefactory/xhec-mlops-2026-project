"""Check target isolation and the saved training/inference pipeline."""

import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.modelling.deployment import serve_daily_training
from src.modelling.main import main, train_abalone_model
from src.modelling.predicting import predict
from src.modelling.preprocessing import split_features_target
from src.modelling.training import train_model
from src.modelling.utils import pickle_object


class ModellingTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "Sex": ["M", "F", "I"] * 10,
                "Length": np.linspace(0.1, 0.8, 30),
                "Rings": np.arange(30) % 12 + 2,
            }
        )

    def test_target_excluded(self):
        data = self.data.assign(Age=self.data.Rings + 1.5)
        features, target = split_features_target(data)
        self.assertEqual(list(features.columns), ["Sex", "Length"])
        np.testing.assert_allclose(target, data.Rings + 1.5)
        with self.assertRaisesRegex(ValueError, "Rings"):
            split_features_target(features)

    def test_pickle_roundtrip_and_unseen_category(self):
        features, target = split_features_target(self.data)
        model = train_model(features, target)
        unseen = features.iloc[:2].copy()
        unseen["Sex"] = "unknown"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects/model.pkl"
            pickle_object(model, path)
            np.testing.assert_allclose(predict(path, unseen), model.predict(unseen))

    def test_notebook_baseline_parity_and_training_only_scaling(self):
        """Keep the notebook baseline and prevent fitting scaling on test rows."""
        features, target = split_features_target(self.data)
        x_train, x_test, y_train, _ = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        reference = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
                            ("num", StandardScaler(), ["Length"]),
                        ]
                    ),
                ),
                ("model", LinearRegression()),
            ]
        ).fit(x_train, y_train)
        model = train_model(x_train, y_train)
        np.testing.assert_allclose(model.predict(x_test), reference.predict(x_test))
        scaler = model.named_steps["preprocess"].named_transformers_["num"]
        np.testing.assert_allclose(scaler.mean_, x_train[["Length"]].mean())

    def test_training_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "abalone.csv"
            model_path = Path(directory) / "model.pkl"
            self.data.to_csv(csv_path, index=False)
            main(csv_path, model_path)
            features, _ = split_features_target(self.data)
            predictions = predict(model_path, features)
            self.assertEqual(len(predictions), len(features))
            self.assertTrue(np.isfinite(predictions).all())

    def test_prefect_training_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "abalone.csv"
            model_path = Path(directory) / "model.pkl"
            self.data.to_csv(csv_path, index=False)

            metrics = train_abalone_model(csv_path, model_path)

            self.assertEqual(set(metrics), {"rmse", "mae", "r2"})
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertTrue(model_path.is_file())

    def test_daily_deployment_schedule(self):
        with patch("src.modelling.deployment.train_abalone_model.serve") as serve:
            serve_daily_training()

        arguments = serve.call_args.kwargs
        self.assertEqual(arguments["name"], "daily-abalone-retraining")
        self.assertEqual(arguments["schedule"].cron, "0 2 * * *")
        self.assertEqual(arguments["schedule"].timezone, "UTC")
        self.assertEqual(
            arguments["parameters"]["trainset_path"],
            str(Path("data/abalone.csv").resolve()),
        )

    def test_command_line_training_and_prediction(self):
        """A separate Python process can load the artifact and predict."""
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "abalone.csv"
            model_path = Path(directory) / "model.pkl"
            self.data.to_csv(csv_path, index=False)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.modelling.main",
                    str(csv_path),
                    "--model-path",
                    str(model_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.modelling.predicting",
                    str(model_path),
                    str(csv_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output = pd.read_csv(io.StringIO(result.stdout))
            self.assertEqual(list(output.columns), ["Age"])
            self.assertEqual(len(output), len(self.data))
            self.assertTrue(np.isfinite(output.Age).all())
