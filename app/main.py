from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="ML API for predicting Titanic passenger survival",
    version="1.0.0"
)


# Load model
MODEL_PATH = "models/model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# Request schema
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(passenger: Passenger):

    data = pd.DataFrame([passenger.model_dump()])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return {
        "survived": int(prediction),
        "survival_probability": round(float(probability), 4)
    }