from pathlib import Path
import pandas as pd


UNUSED_COLS = ["Unnamed: 0"]
GENDER_MAPPING = {"Male": 0, "Female": 1, "1.0": 2, "0.0": 3}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in UNUSED_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(GENDER_MAPPING)
    return df


def prepare_input(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    df = encode_gender(df)
    return df

