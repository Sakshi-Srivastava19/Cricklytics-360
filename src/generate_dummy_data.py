# src/generate_dummy_data.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

os.makedirs("data", exist_ok=True)

teams = ["CSK","MI","RCB","KKR","PBKS","SRH","RR","LSG"]
players = [f"Player_{i}" for i in range(1,101)]

rows = []
start = datetime(2017,1,1)
for i in range(1200):  # ~1200 matches
    date = start + timedelta(days=i)
    t1, t2 = random.sample(teams, 2)
    venue = random.choice(["StadiumA", "StadiumB", "StadiumC"])
    # features (toy)
    team1_score = random.randint(100,220)
    team2_score = team1_score + random.randint(-50,50)
    match_result = 1 if team1_score > team2_score else 0  # 1 means team1 win
    toss_winner = random.choice([t1, t2])
    toss_decision = random.choice(["bat","field"])
    # simplified features
    form_team1 = random.uniform(0,1)
    form_team2 = random.uniform(0,1)
    avg_player_rating = random.uniform(0,1)
    rows.append({
        "match_id": f"M{i+1}",
        "date": date.strftime("%Y-%m-%d"),
        "team1": t1,
        "team2": t2,
        "venue": venue,
        "team1_score": team1_score,
        "team2_score": team2_score,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "form_team1": round(form_team1,3),
        "form_team2": round(form_team2,3),
        "avg_player_rating": round(avg_player_rating,3),
        "team1_win": match_result
    })

df = pd.DataFrame(rows)
df.to_csv("data/dummy_matches.csv", index=False)
print("Saved data/dummy_matches.csv with", len(df), "rows")
