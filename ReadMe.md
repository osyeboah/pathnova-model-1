# Pathnova AI – Model 1: Goal Feasibility Predictor

This project contains the full training pipeline for **Model 1** of the Pathnova AI system — the **Goal Feasibility Predictor**.

It evaluates how feasible a user’s income or career goal is, based on features like current salary, target salary, experience, skills, education, and timeframe.

---
# Folder Structure
pathnova-ai/
│
├── data/
│ └── goal_feasibility.csv # Synthetic training dataset
│
├── ml/
│ └── training/
│ └── train_goal_feasibility.py # Model training script
│
├── models/
│ └── goal_feasibility/
│ ├── xgb_model.joblib # Saved XGBoost model
│ └── preprocessing.pkl # Saved StandardScaler
│
├── services/
│ └── ai/
│ └── goal_engine/
│ └── goal_feasibility.py # (FastAPI service coming soon)
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## Technologies Used

- Python 3.10+
- XGBoost (model)
- Scikit-learn (preprocessing)
- Pandas
- Joblib
- FastAPI *(WIP)*

---

## How to Train the Model

To train the model from scratch:

```bash
python ml/training/train_goal_feasibility.py
## After deployment, the model will return a prediction like
{
  "feasibility_score": 0.87,
  "status": "Feasible"
}

