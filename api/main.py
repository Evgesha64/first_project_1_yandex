from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.predictor import Predictor
from ml.data_loader import prepare_input


MODEL_PATH = Path("artifacts/model_pipeline.pkl")

app = FastAPI(title="Heart Risk Prediction API")
predictor = Predictor(MODEL_PATH)
predictor.load()


class PredictRequest(BaseModel):
    csv_path: str


class PredictionItem(BaseModel):
    id: int
    prediction: int


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    csv_path = Path(request.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail="CSV file not found.")

    df = prepare_input(csv_path)
    if "id" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'id' column.")

    preds_df = predictor.predict_df(df)
    items = [PredictionItem(id=int(row.id), prediction=int(row.prediction)) for row in preds_df.itertuples(index=False)]
    return PredictResponse(predictions=items)

