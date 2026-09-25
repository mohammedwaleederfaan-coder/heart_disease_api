from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import joblib
import pandas as pd

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler('.log')
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    logger.error("File Not Found please download it")
    model = None
except Exception as e:
    logger.error(f"An error occured while loading the model: {e}")
    model = None

class Features(BaseModel):
    Age: int = Field(..., gt=0, lt=100)
    Sex: str = Field(...)
    ChestPainType: str = Field(...)
    RestingBP: int = Field(...)
    Cholesterol: int = Field(...)
    FastingBS: int = Field(..., ge=0, le=1)
    RestingECG: str = Field(...)
    MaxHR: int = Field(..., gt=59)
    ExerciseAngina: str = Field(...)
    Oldpeak: float = Field(..., ge=0)
    ST_Slope: str = Field(...)


@app.get('/')
def home():
    return {'message': 'Server Running'}

@app.post("/predict")
def predict(features: Features):

    if model is None:
        logger.error("Model is not loaded")
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    try:
        df = pd.DataFrame([features.model_dump()])
        prediction = model.predict(df)[0]

        return {
            "prediction": int(prediction)
        }

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )