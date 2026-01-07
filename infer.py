import argparse
from pathlib import Path

import pandas as pd

from ml.predictor import Predictor


def run_inference(model_path: Path, input_csv: Path, output_csv: Path) -> None:
    predictor = Predictor(model_path)
    pred_df = predictor.predict_file(input_csv)

    # Сохраняем с индексом, чтобы test.py (index_col=0) прочитал корректно
    pred_df.to_csv(output_csv, index=True)
    print(f"Saved predictions to {output_csv}")
    print(pred_df.head())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and save predictions CSV.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model_pipeline.pkl"),
        help="Path to saved model pipeline.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/heart_test.csv"),
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("student_predictions.csv"),
        help="Path to save predictions CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model, args.input, args.output)

