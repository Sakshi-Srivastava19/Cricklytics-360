# src/prepare_dataset.py
import pandas as pd
import os

def build_match_level(matches_path="data/matches.csv", deliveries_path="data/deliveries.csv", out_path="data/processed_matches.csv"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # aggregate runs per innings per match
    agg = deliveries.groupby(['match_id', 'inning']).agg({
        'total_runs': 'sum',
        'player_dismissed': lambda x: x.notna().sum()
    }).reset_index().rename(columns={'total_runs': 'runs', 'player_dismissed': 'wickets'})

    # pivot innings to team1/team2 (inning 1 -> team1, inning 2 -> team2)
    # map match_id to team names from matches file
    matches_small = matches[['id','team1','team2','venue','toss_winner','toss_decision','winner','season','date']].rename(columns={'id':'match_id'})
    # join innings 1 and 2
    in1 = agg[agg['inning']==1][['match_id','runs','wickets']].rename(columns={'runs':'runs1','wickets':'wickets1'})
    in2 = agg[agg['inning']==2][['match_id','runs','wickets']].rename(columns={'runs':'runs2','wickets':'wickets2'})

    df = matches_small.merge(in1, on='match_id', how='left').merge(in2, on='match_id', how='left')

    # Fill NaNs for incomplete matches
    df['runs1'] = df['runs1'].fillna(0).astype(int)
    df['runs2'] = df['runs2'].fillna(0).astype(int)
    df['wickets1'] = df['wickets1'].fillna(0).astype(int)
    df['wickets2'] = df['wickets2'].fillna(0).astype(int)

    # Create binary target: team1_win (1 if team1 is winner else 0)
    df['team1_win'] = (df['winner'] == df['team1']).astype(int)
    # For modeling, create canonical "winner" column as 1/0 where 1 = team1 wins
    df['winner_bin'] = df['team1_win']

    # Add simple recent form: compute last 3-match win rate for each team
    df = df.sort_values(['season','date']).reset_index(drop=True)
    team_history = {}
    recent_winrate = []
    for idx, row in df.iterrows():
        t1, t2 = row['team1'], row['team2']
        # get last 3 results
        def winrate(team):
            history = team_history.get(team, [])
            if len(history)==0: return 0.5
            return sum(history[-3:]) / max(1, min(3, len(history)))
        recent_winrate.append((winrate(t1), winrate(t2)))
        # after computing, append current result to history
        # team1 result: 1 if team1 won
        team_history.setdefault(t1, []).append(int(row['team1_win']))
        team_history.setdefault(t2, []).append(int(1-row['team1_win']))  # for team2, win if team1 lost

    df['form_team1'] = [x[0] for x in recent_winrate]
    df['form_team2'] = [x[1] for x in recent_winrate]

    # Keep columns we need
    processed = df[['match_id','season','date','team1','team2','venue','toss_winner','toss_decision',
                    'runs1','runs2','wickets1','wickets2','form_team1','form_team2','winner_bin']]
    processed = processed.rename(columns={'runs1':'team1_runs','runs2':'team2_runs','wickets1':'team1_wickets','wickets2':'team2_wickets','winner_bin':'team1_win'})

    processed.to_csv(out_path, index=False)
    print("Saved", out_path, "shape:", processed.shape)
    return processed

if __name__ == "__main__":
    build_match_level()
