from pathlib import Path
from typing import List
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ml.predictor import Predictor
from ml.data_loader import prepare_input


MODEL_PATH = Path("artifacts/model_pipeline.pkl")

app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API для предсказания риска сердечного приступа на основе данных пациентов",
    version="1.0.0"
)

# Загрузка модели при старте
predictor = Predictor(MODEL_PATH)
predictor.load()


class PredictRequest(BaseModel):
    csv_path: str


class PredictionItem(BaseModel):
    id: int
    prediction: int


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Веб-интерфейс для загрузки файлов и просмотра результатов"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Heart Attack Risk Prediction API</h1><p>Веб-интерфейс не найден</p>"


@app.post("/predict", response_model=PredictResponse)
def predict_from_path(request: PredictRequest) -> PredictResponse:
    """
    Предсказание по пути к CSV файлу
    
    - **csv_path**: путь к CSV файлу с данными пациентов
    """
    csv_path = Path(request.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV файл не найден: {csv_path}")
    
    try:
        df = prepare_input(csv_path)
        if "id" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV должен содержать колонку 'id'")
        
        preds_df = predictor.predict_df(df)
        items = [
            PredictionItem(id=int(row.id), prediction=int(row.prediction))
            for row in preds_df.itertuples(index=False)
        ]
        return PredictResponse(predictions=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_from_upload(file: UploadFile = File(...)) -> PredictResponse:
    """
    Предсказание по загруженному CSV файлу
    
    - **file**: CSV файл с данными пациентов
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    
    try:
        # Сохраняем загруженный файл во временный файл
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        try:
            df = prepare_input(tmp_path)
            if "id" not in df.columns:
                raise HTTPException(status_code=400, detail="CSV должен содержать колонку 'id'")
            
            preds_df = predictor.predict_df(df)
            items = [
                PredictionItem(id=int(row.id), prediction=int(row.prediction))
                for row in preds_df.itertuples(index=False)
            ]
            return PredictResponse(predictions=items)
        finally:
            # Удаляем временный файл
            if tmp_path.exists():
                tmp_path.unlink()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")


@app.get("/health")
def health_check():
    """Проверка работоспособности API"""
    return {"status": "ok", "model_loaded": predictor.pipeline is not None}
