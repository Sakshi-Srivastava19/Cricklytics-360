# src/shap_tools.py
import shap
import pickle
import pandas as pd
import numpy as np

def load_model_and_encoder(model_path="models/random_forest_model.pkl", encoder_path="models/encoder.pkl"):
    model = pickle.load(open(model_path, "rb"))
    encoder = pickle.load(open(encoder_path, "rb"))
    return model, encoder

def compute_shap_df(model, encoder, df_sample):
    """
    df_sample: DataFrame with columns used for encoder: team1,team2,venue,toss_decision
               and numeric cols matching training features.
    Returns shap values and base values and combined DataFrame for display.
    """
    cat_cols = list(encoder.feature_names_in_) if hasattr(encoder, 'feature_names_in_') else []
    # Build X as model expects
    X_cat = encoder.transform(df_sample[['team1','team2','venue','toss_decision']])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out())
    numeric_cols = [c for c in df_sample.columns if c not in ['team1','team2','venue','toss_decision']]
    X = pd.concat([X_cat_df.reset_index(drop=True), df_sample[numeric_cols].reset_index(drop=True)], axis=1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # list for multi-class; for binary returns array
    # For binary classifier shap_values is a list with two arrays; use shap_values[1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    return sv, X, explainer.expected_value if hasattr(explainer, 'expected_value') else explainer.expected_value
