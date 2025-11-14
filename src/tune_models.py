# src/tune_models.py
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def load_processed(path="data/processed_matches.csv"):
    return pd.read_csv(path)

def build_X_y(df):
    cat = ['team1','team2','venue','toss_decision']
    num = ['team1_runs','team2_runs','team1_wickets','team2_wickets','form_team1','form_team2']
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = enc.fit_transform(df[cat])
    X_cat_df = pd.DataFrame(X_cat, columns=enc.get_feature_names_out(cat))
    X = pd.concat([X_cat_df, df[num].reset_index(drop=True)], axis=1)
    y = df['team1_win'].astype(int)
    return X, y, enc

def tune_rf(X_train, y_train, n_iter=20, cv=3, random_state=42):
    param_dist = {
        'n_estimators': [100,200,400,700],
        'max_depth': [None, 6, 10, 20],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4]
    }
    rf = RandomForestClassifier(random_state=random_state)
    rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=cv, n_jobs=-1, verbose=1, random_state=random_state)
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_, rs.best_score_

def tune_xgb(X_train, y_train, n_iter=20, cv=3, random_state=42):
    param_dist = {
        'n_estimators': [100,200,400],
        'max_depth': [3,6,10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6,0.8,1.0],
        'colsample_bytree': [0.6,0.8,1.0]
    }
    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=random_state)
    rs = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=n_iter, cv=cv, n_jobs=-1, verbose=1, random_state=random_state)
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_, rs.best_score_

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    df = load_processed("data/processed_matches.csv")
    X, y, encoder = build_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Tuning Random Forest...")
    best_rf, rf_params, rf_score = tune_rf(X_train, y_train, n_iter=30)
    print("RF best score (cv):", rf_score)
    # evaluate on test
    rf_test_acc = accuracy_score(y_test, best_rf.predict(X_test))
    print("RF test acc:", rf_test_acc)

    print("Tuning XGBoost...")
    best_xgb, xgb_params, xgb_score = tune_xgb(X_train, y_train, n_iter=30)
    xgb_test_acc = accuracy_score(y_test, best_xgb.predict(X_test))
    print("XGB test acc:", xgb_test_acc)

    # Save tuned models & encoder
    with open("models/encoder.pkl","wb") as f:
        pickle.dump(encoder, f)
    with open("models/tuned_rf_model.pkl","wb") as f:
        pickle.dump(best_rf, f)
    with open("models/tuned_xgb_model.pkl","wb") as f:
        pickle.dump(best_xgb, f)

    # Save best params
    pd.DataFrame([{"model":"random_forest","cv_score":rf_score,"test_acc":rf_test_acc,"params":rf_params},
                  {"model":"xgboost","cv_score":xgb_score,"test_acc":xgb_test_acc,"params":xgb_params}]).to_csv("models/tuning_results.csv", index=False)

    print("Tuning done. Saved tuned_rf_model.pkl and tuned_xgb_model.pkl")
