from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from ml.data_loader import prepare_input
# Ensure SkCatBoostClassifier is imported so joblib can load the class
from ml.model_wrappers import SkCatBoostClassifier  # noqa: F401


class Predictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = None

    def load(self) -> None:
        if self.pipeline is None:
            artifacts = joblib.load(self.model_path)
            self.imputer = artifacts["imputer"]
            self.model = artifacts["model"]
            self.feature_cols = artifacts["feature_cols"]
            self.pipeline = True  # marker

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded. Call load() first.")

        X = df[self.feature_cols]
        X_proc = self.imputer.transform(X)
        preds = self.model.predict(X_proc)
        result = pd.DataFrame(
            {
                "id": df["id"].values,
                "prediction": preds.astype(int),
            }
        )
        return result

    def predict_file(self, csv_path: Path) -> pd.DataFrame:
        self.load()
        df = prepare_input(csv_path)
        return self.predict_df(df)

