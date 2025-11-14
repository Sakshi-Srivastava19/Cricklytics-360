# predict.py
import pickle
import pandas as pd
import numpy as np

# load best model (choose random_forest or xgboost based on model_scores.csv)
model = pickle.load(open("models/random_forest_model.pkl","rb"))
encoder = pickle.load(open("models/encoder.pkl","rb"))

def predict_match(team1, team2, venue, toss_decision, team1_runs, team2_runs, team1_wickets, team2_wickets, form1, form2):
    df = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_decision": toss_decision,
        "team1_runs": team1_runs,
        "team2_runs": team2_runs,
        "team1_wickets": team1_wickets,
        "team2_wickets": team2_wickets,
        "form_team1": form1,
        "form_team2": form2
    }])
    X_cat = encoder.transform(df[["team1","team2","venue","toss_decision"]])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out())
    X_num = df[["team1_runs","team2_runs","team1_wickets","team2_wickets","form_team1","form_team2"]]
    X = pd.concat([X_cat_df, X_num], axis=1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    winner = team1 if pred==1 else team2
    confidence = max(proba)
    return winner, confidence

if __name__ == "__main__":
    # example
    w,c = predict_match("CSK","MI","Chennai","bat",160,150,5,7,0.6,0.4)
    print("Predicted:", w, "Confidence:", round(c*100,2), "%")
