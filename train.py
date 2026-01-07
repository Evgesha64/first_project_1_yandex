import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from ml.model_wrappers import SkCatBoostClassifier

RANDOM_STATE = 42
TARGET_COL = "Heart Attack Risk (Binary)"
ID_COL = "id"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.pkl"


def load_data(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def encode_gender(series: pd.Series) -> pd.Series:
    mapping = {"Male": 0, "Female": 1, "1.0": 2, "0.0": 3}
    return series.map(mapping)


def train(train_csv: Path) -> None:
    df = load_data(train_csv)
    if "Gender" in df.columns:
        df["Gender"] = encode_gender(df["Gender"])

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    imputer = SimpleImputer(strategy="median")
    X_proc = imputer.fit_transform(X)

    model = SkCatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=1,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=RANDOM_STATE,
        verbose=False,
        class_weights=[1.0, 1.88],
    )
    model.fit(X_proc, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"imputer": imputer, "model": model, "feature_cols": feature_cols},
        MODEL_PATH,
    )
    print(f"Saved imputer+model to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CatBoost model and save pipeline.")
    parser.add_argument("--train", type=Path, default=Path("datasets/heart_train.csv"))
    args = parser.parse_args()
    train(args.train)