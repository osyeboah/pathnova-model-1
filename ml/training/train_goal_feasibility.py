print("🚀 Starting training script...")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os

# Step 1: Load the dataset
try:
    df = pd.read_csv('data/goal_feasibility.csv')
    print(f"📊 Loaded dataset with shape: {df.shape}")
except FileNotFoundError:
    print("❌ Dataset not found. Make sure goal_feasibility.csv is in the data/ folder.")
    exit()

# Step 2: Split features and target
X = df.drop(columns=['feasible'])   # Input features
y = df['feasible']                  # Target label

# Step 3: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("🔧 Features scaled.")

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("📦 Data split into training and test sets.")

# Step 5: Train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
print("✅ XGBoost model trained.")

# Step 6: Save model and scaler
os.makedirs('models/goal_feasibility', exist_ok=True)
joblib.dump(model, 'models/goal_feasibility/xgb_model.joblib')
joblib.dump(scaler, 'models/goal_feasibility/preprocessing.pkl')

print("💾 Model and scaler saved to 'models/goal_feasibility/'")
print("✅ Training pipeline complete.")




