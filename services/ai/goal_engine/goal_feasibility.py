from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/goal_feasibility/xgb_model.joblib")
scaler = joblib.load("models/goal_feasibility/preprocessing.pkl")

app = FastAPI(title="Goal Feasibility Predictor")

class PredictionInput(BaseModel):
    current_salary: float
    target_salary: float
    experience_years: float
    skill_score: float
    education_level: int
    goal_timeframe_months: int

@app.get("/")
def read_root():
    return {"message": "Goal Feasibility API is running "}

@app.post("/predict")
def predict(input: PredictionInput):
    features = np.array([[
        input.current_salary,
        input.target_salary,
        input.experience_years,
        input.skill_score,
        input.education_level,
        input.goal_timeframe_months
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return {"feasible": int(prediction[0])}

