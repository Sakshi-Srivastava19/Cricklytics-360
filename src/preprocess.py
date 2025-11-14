# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_processed(path="data/processed_matches.csv"):
    df = pd.read_csv(path)
    return df

def prepare_features(df, categorical_cols=None, numeric_cols=None, target_col='team1_win'):
    # default features
    if categorical_cols is None:
        categorical_cols = ['team1','team2','venue','toss_decision']
    if numeric_cols is None:
        numeric_cols = ['team1_runs','team2_runs','team1_wickets','team2_wickets','form_team1','form_team2']

    # make sure columns exist
    for c in categorical_cols + numeric_cols + [target_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = enc.fit_transform(df[categorical_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=enc.get_feature_names_out(categorical_cols))

    X_num = df[numeric_cols].reset_index(drop=True)
    X = pd.concat([X_cat_df, X_num], axis=1)
    X.columns = X.columns.astype(str)  # sklearn requires str names

    y = df[target_col].astype(int)
    return X, y, enc
