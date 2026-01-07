# Heart Attack Risk Prediction

Модель и сервис для предсказания риска сердечного приступа по данным пациентов.

## Этапы разработки
1. Исследование и обучение в `Projekt.ipynb`: EDA, подбор параметров CatBoost, финальная модель (F1≈0.77).
2. Экспорт пайплайна в код: медианный impute + CatBoost с финальными гиперпараметрами.
3. CLI-скрипты для обучения (`train.py`) и инференса (`infer.py`).
4. FastAPI-сервис `/predict` для интеграции.
5. Формат предсказаний совместим с тестовым скриптом `test.py`.

## Структура
- `datasets/heart_train.csv`, `datasets/heart_test.csv`
- `train.py` — обучение модели и сохранение артефакта
- `infer.py` — инференс и генерация `student_predictions.csv`
- `ml/` — загрузка/кодирование и класс `Predictor`
- `api/main.py` — FastAPI API (`/predict`)
- `artifacts/model_pipeline.pkl` — имьютер + модель + список фичей (после обучения)
- `test.py` — проверка предсказаний против `correct_answers.csv`
- `Projekt.ipynb` — полный EDA и первичное обучение

## Установка
```bash
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt
```

## Обучение модели
```bash
.\.venv\Scripts\python train.py --train datasets/heart_train.csv
```
Сохранится `artifacts/model_pipeline.pkl`.

## Инференс
```bash
.\.venv\Scripts\python infer.py --model artifacts/model_pipeline.pkl --input datasets/heart_test.csv --output student_predictions.csv
```
Результат: CSV с индексом и колонками `id, prediction` (классы 0/1).

## Проверка (если есть correct_answers.csv)
```bash
.\.venv\Scripts\python test.py --student student_predictions.csv --correct correct_answers.csv
```

## Запуск API
```bash
.\.venv\Scripts\python -m uvicorn api.main:app --reload
```
После запуска откройте в браузере: http://127.0.0.1:8000

### Веб-интерфейс
- **GET /** — веб-интерфейс для загрузки CSV файлов и просмотра результатов
- Поддержка drag & drop файлов
- Отображение результатов в таблице

### API эндпоинты

#### 1. Предсказание по пути к файлу
```http
POST /predict
Content-Type: application/json

{
  "csv_path": "datasets/heart_test.csv"
}
```
Ответ: `{"predictions":[{"id":..., "prediction":...}, ...]}`

#### 2. Предсказание по загруженному файлу
```http
POST /predict/upload
Content-Type: multipart/form-data

file: [CSV файл]
```
Ответ: `{"predictions":[{"id":..., "prediction":...}, ...]}`

#### 3. Проверка работоспособности
```http
GET /health
```
Ответ: `{"status": "ok", "model_loaded": true}`

## Формат данных и обработка
- `Unnamed: 0` удаляется.
- `Gender` кодируется: `{'Male':0,'Female':1,'1.0':2,'0.0':3}`.
- Пропуски — медиана по train.
- Таргет: `Heart Attack Risk (Binary)`, `prediction` — класс 0/1.

## Совместимость с test.py
- `student_predictions.csv` должен содержать индекс и колонки `id`, `prediction`, длина = тест.
- Скрипт проверки: `python test.py --student ... --correct correct_answers.csv`.

## Зависимости (requirements.txt)
- catboost==1.2.8
- fastapi==0.115.5
- uvicorn==0.32.1
- scikit-learn==1.5.2
- pandas==2.3.3
- numpy==2.4.0
- joblib==1.5.3

