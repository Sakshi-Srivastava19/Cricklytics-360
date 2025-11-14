# src/data_loader.py
import pandas as pd

def load_matches(path="data/dummy_matches.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    # quick cleanup
    df = df.dropna()
    return df
