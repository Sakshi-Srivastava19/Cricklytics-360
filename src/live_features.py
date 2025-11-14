# src/live_features.py
import pandas as pd
import numpy as np

def load_deliveries(path="data/deliveries.csv"):
    return pd.read_csv(path)

def compute_live_features(deliveries, match_id, inning=1, upto_over=10.0):
    """
    deliveries: deliveries dataframe
    match_id: match id (must match deliveries['match_id'] type)
    inning: 1 or 2
    upto_over: float like 10.0 means up to 10 overs (i.e., over <= 10)
    """
    # filter
    d = deliveries[(deliveries['match_id'] == match_id) & (deliveries['inning'] == inning)].copy()
    # Over column in dataset may be 'over' or 'over_number' - try 'over'
    if 'over' not in d.columns:
        # fallback column names
        if 'ball' in d.columns:
            # can't compute overs reliably, assume deliveries have 'over' column
            raise ValueError("deliveries must include an 'over' column")
        else:
            raise ValueError("deliveries missing expected 'over' column")
    d = d[d['over'] <= upto_over]

    # basic features
    runs = d['total_runs'].sum() if 'total_runs' in d.columns else d['runs'].sum()
    wickets = d['player_dismissed'].notna().sum()
    balls = int(len(d))
    overs_completed = upto_over
    # last 5 overs runs
    last5_start = upto_over - 5
    if last5_start < 0: last5_start = 0
    last5 = d[d['over'] > last5_start]['total_runs'].sum() if 'total_runs' in d.columns else d[d['over'] > last5_start]['runs'].sum()

    # strike rate / run rate
    run_rate = runs / overs_completed if overs_completed > 0 else 0.0
    return {
        "current_score": int(runs),
        "wickets_left": max(0, 10 - int(wickets)),
        "overs_done": float(overs_completed),
        "runs_last_5": int(last5),
        "balls": balls,
        "run_rate": run_rate
    }

# Example usage:
# deliveries = load_deliveries("data/deliveries.csv")
# features = compute_live_features(deliveries, match_id=1, inning=1, upto_over=10.0)
# print(features)
