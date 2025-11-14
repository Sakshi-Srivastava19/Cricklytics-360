# train.py
import os, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocess import load_processed, prepare_features

def main():
    os.makedirs("models", exist_ok=True)
    df = load_processed("data/processed_matches.csv")
    X, y, encoder = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgboost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    }

    results = []
    trained = {}
    for name, model in models.items():
        print("Training", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(name, "accuracy:", acc)
        print(classification_report(y_test, preds))
        results.append((name, acc))
        trained[name] = model

    # Save encoder and each model
    with open("models/encoder.pkl","wb") as f:
        pickle.dump(encoder, f)
    for name, model in trained.items():
        with open(f"models/{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)

    # save results for dashboard
    pd.DataFrame(results, columns=['Model','Accuracy']).to_csv("models/model_scores.csv", index=False)
    best = max(results, key=lambda x: x[1])
    print("Best model:", best)

if __name__ == "__main__":
    main()
