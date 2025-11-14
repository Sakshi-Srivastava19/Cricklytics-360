# src/models.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_rf(X_train, y_train, X_val, y_val):
    print("Training RandomForest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print("Validation Accuracy:", acc)
    print(classification_report(y_val, preds))
    joblib.dump(model, f"{MODEL_DIR}/rf_model.pkl")
    return model

def train_xgb(X_train, y_train, X_val, y_val):
    print("Training XGBoost...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print("Validation Accuracy:", acc)
    joblib.dump(model, f"{MODEL_DIR}/xgb_model.pkl")
    return model

def load_model(path="models/rf_model.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"Model not found at {path}")
