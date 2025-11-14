import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import shap
import numpy as np
from openai import OpenAI
import shap
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Cricklytics 360", page_icon="🏏", layout="wide")

# =========================
# SESSION STATE (for navigation)
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page_name):
    st.session_state.page = page_name

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("🏏 Cricklytics 360 Navigation")
st.sidebar.button("🏠 Home", on_click=lambda: go_to("home"))
st.sidebar.button("🔮 Live Match Prediction", on_click=lambda: go_to("predict"))
st.sidebar.button("📊 Model Comparison", on_click=lambda: go_to("compare"))
st.sidebar.button("📈 Insights", on_click=lambda: go_to("insights"))
st.sidebar.button("💬 Chatbot Assistant", on_click=lambda: go_to("💬 Chatbot Assistant"))

# =========================
# PAGE 1: HOME
# =========================
if  st.session_state.page == "home":
    st.title("🏏 Cricklytics 360")
    st.markdown("""
    ### Welcome to **Cricklytics 360** — Your AI-driven Cricket Analysis Platform!  
    🧠 Predict match outcomes, 📊 Compare ML models, and 📈 Explore IPL insights — all in one place.
    """)

    st.image("assets/image.png", width='stretch',)

    st.markdown("### Choose an Option Below 👇")

    col1, col2, col3,col4 = st.columns(4)

    with col1:
        if st.button("🎯 Match Prediction"):
            st.session_state.page = "predict"
            st.rerun()

    with col2:
        if st.button("🤖 Model Comparison"):
            st.session_state.page = "compare"
            st.rerun()

    with col3:
        if st.button("📊 Data Insights"):
            st.session_state.page = "insights"
            st.rerun()
    with col4:
        if st.button("💬 Chatbot Assistant"):
            st.session_state.page = "💬 Chatbot Assistant"
            st.rerun()
# =========================
# PAGE 2: LIVE MATCH PREDICTION
# =========================
elif st.session_state.page == "predict":
    st.title("🔮 Live Match Prediction")

    # Load models
    model_path = "models/random_forest_model.pkl"
    encoder_path = "models/encoder.pkl"

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        st.error("⚠️ Models not found. Please train the models first by running train.py.")
    else:
        model = pickle.load(open(model_path, "rb"))
        encoder = pickle.load(open(encoder_path, "rb"))

        df_proc = pd.read_csv("data/processed_matches.csv")
        teams = sorted(set(df_proc['team1']).union(set(df_proc['team2'])))
        venues = sorted(df_proc['venue'].unique())

        st.sidebar.header("Match Inputs")

        team1 = st.sidebar.selectbox("Team 1", teams)
        team2 = st.sidebar.selectbox("Team 2", [t for t in teams if t != team1])
        venue = st.sidebar.selectbox("Venue", venues)
        toss = st.sidebar.selectbox("Toss Decision", ["bat", "field"])
        team1_runs = st.sidebar.number_input("Team 1 runs", 0, 400, 150)
        team2_runs = st.sidebar.number_input("Team 2 runs", 0, 400, 140)
        team1_wickets = st.sidebar.number_input("Team 1 wickets", 0, 10, 6)
        team2_wickets = st.sidebar.number_input("Team 2 wickets", 0, 10, 8)
        form1 = st.sidebar.slider("Team 1 recent form", 0.0, 1.0, 0.5)
        form2 = st.sidebar.slider("Team 2 recent form", 0.0, 1.0, 0.5)

        if st.button("Predict Winner 🏆"):
            try:
                input_df = pd.DataFrame([{
                    "team1": team1,
                    "team2": team2,
                    "venue": venue,
                    "toss_decision": toss,
                    "team1_runs": team1_runs,
                    "team2_runs": team2_runs,
                    "team1_wickets": team1_wickets,
                    "team2_wickets": team2_wickets,
                    "form_team1": form1,
                    "form_team2": form2
                }])

                # Encode
                X_cat = encoder.transform(input_df[["team1","team2","venue","toss_decision"]])
                X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out())
                X_num = input_df[["team1_runs","team2_runs","team1_wickets","team2_wickets","form_team1","form_team2"]]
                X = pd.concat([X_cat_df, X_num], axis=1)

                # Predict
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]

                winner = team1 if pred == 1 else team2
                st.success(f"🏆 Predicted Winner: {winner}")
                st.progress(float(max(proba)))

                st.markdown(f"**Confidence:** {round(max(proba) * 100, 2)}%")

                # SHAP visualization
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                st.subheader("📉 Feature Importance (SHAP)")
                st.bar_chart(pd.DataFrame(shap_values[0], index=X.columns))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# =========================
# PAGE 3: MODEL COMPARISON
# =========================
elif st.session_state.page == "compare":
    st.title("📊 Model Comparison Dashboard")

    if os.path.exists("models/model_scores.csv"):
        results_df = pd.read_csv("models/model_scores.csv")
        st.dataframe(results_df, use_container_width=True)
        best_model = results_df.loc[results_df["Accuracy"].idxmax()]
        st.success(f"🏆 Best Model: {best_model['Model']} ({best_model['Accuracy']*100:.2f}% accuracy)")
        st.bar_chart(results_df.set_index("Model")["Accuracy"])
    else:
        st.warning("No model performance data found. Please train models first.")

# =========================
# PAGE 4: INSIGHTS (IMPROVED VERSION)
# =========================
elif st.session_state.page == "insights":
    st.title("📈 IPL Advanced Insights Dashboard")

    if not os.path.exists("data/processed_matches.csv"):
        st.error("⚠️ Processed data not found. Please ensure data/processed_matches.csv exists.")
    else:
        df = pd.read_csv("data/processed_matches.csv")

        st.subheader("🔍 Dataset Overview")
        st.write(df.head())

        # ----------------------------------------------------
        # SAFE COLUMN HANDLING (Fixes KeyError: team_1)
        # ----------------------------------------------------
        team_column_pairs = [
            ("team_1", "team_2"),
            ("team1", "team2"),
            ("bat_team", "bowl_team"),
            ("home_team", "away_team")
        ]

        found_pair = None
        for c1, c2 in team_column_pairs:
            if c1 in df.columns and c2 in df.columns:
                found_pair = (c1, c2)
                break

        if found_pair:
            df.rename(columns={found_pair[0]: "team_1", found_pair[1]: "team_2"}, inplace=True)
        else:
            if "team" in df.columns:
                df["team_1"] = df["team"]
                df["team_2"] = df["team"]
            else:
                st.error("❌ No team columns found in dataset.")
                st.stop()

        # Fix winner column
        if "winner" not in df.columns:
            for alt in ["winning_team", "match_winner", "result"]:
                if alt in df.columns:
                    df.rename(columns={alt: "winner"}, inplace=True)
                    break
            if "winner" not in df.columns:
                df["winner"] = "Unknown"

        # Fix season column if needed
        if "season" not in df.columns:
            for alt in ["year", "match_season"]:
                if alt in df.columns:
                    df.rename(columns={alt: "season"}, inplace=True)
                    break

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------
        st.subheader("🎛️ Filters")

        colA, colB = st.columns(2)

        all_teams = sorted(list(set(df["team_1"].unique()) | set(df["team_2"].unique())))
        team_filter = colA.selectbox("Filter by Team", ["All"] + all_teams)

        seasons = sorted(df["season"].dropna().unique()) if "season" in df.columns else []
        season_filter = colB.selectbox("Filter by Season", ["All"] + list(seasons))

        # Apply filters
        filtered_df = df.copy()
        if team_filter != "All":
            filtered_df = filtered_df[
                (filtered_df["team_1"] == team_filter) | (filtered_df["team_2"] == team_filter)
            ]

        if season_filter != "All":
            filtered_df = filtered_df[filtered_df["season"] == season_filter]

        st.markdown("---")

        # ----------------------------------------------------
        # INSIGHTS CHARTS
        # ----------------------------------------------------

        # 1️⃣ Top Venues
        st.subheader("🏟️ Top 10 Venues by Match Count")
        fig1 = px.bar(
            filtered_df["venue"].value_counts().head(10),
            title="Top 10 Venues"
        )
        st.plotly_chart(fig1)

        # 2️⃣ Win Distribution
        st.subheader("🥇 Match Wins by Team")
        if filtered_df["winner"].nunique() > 1:
            fig2 = px.pie(filtered_df, names="winner", title="Win Share")
            st.plotly_chart(fig2)
        else:
            st.info("Not enough data to display win distribution.")

        # 3️⃣ Matches per Season
        if "season" in filtered_df.columns:
            st.subheader("📅 Matches Played per Season")
            fig3 = px.line(
                filtered_df.groupby("season")["match_id"].count(),
                title="Matches per Season"
            )
            st.plotly_chart(fig3)

        # 4️⃣ Toss Winner Impact
        if "toss_winner" in filtered_df.columns:
            st.subheader("🎲 Toss Influence — Does Winning Toss Help?")
            toss_df = filtered_df.groupby("toss_winner")["winner"].count()
            fig4 = px.bar(
                toss_df,
                title="Toss Winner vs Match Winner",
                labels={"value": "Matches Won", "toss_winner": "Team"}
            )
            st.plotly_chart(fig4)

        # 5️⃣ Team Head-to-Head
        st.subheader("⚔️ Team Head-to-Head Performance")
        head_to_head = (
            filtered_df.groupby(["team_1", "team_2"])["winner"].count().reset_index()
        )
        fig5 = px.treemap(
            head_to_head,
            path=["team_1", "team_2"],
            values="winner",
            title="Head-to-Head Treemap"
        )
        st.plotly_chart(fig5)

        # 6️⃣ Win Trends
        st.subheader("📈 Win Trends Over Time")
        win_trend = filtered_df.groupby(["season", "winner"]).size().reset_index(name="wins")
        fig6 = px.line(
            win_trend,
            x="season",
            y="wins",
            color="winner",
            markers=True,
            title="Team Wins Over Seasons"
        )
        st.plotly_chart(fig6)


# CHATBOT ASSISTANT PAGE
elif st.session_state.page == "💬 Chatbot Assistant":
    st.title("💬 Cricket AI Assistant")
    st.markdown("""
    Ask any cricket-related question — stats, predictions, or insights — powered by GPT.
    """)

    # Chatbot UI
    user_input = st.text_input("Type your question here 👇", placeholder="e.g., Who has the most wins at Wankhede?")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_input:
        # Initialize OpenAI client
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

        # Prepare context
        context = "You are Cricklytics AI, an assistant that answers cricket data and prediction questions."

        # Send query
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_input}
            ]
        )

        answer = response.choices[0].message.content
        st.session_state.chat_history.append((user_input, answer))

    # Display chat history
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑‍💻 You:** {q}")
        st.markdown(f"**🤖 Cricklytics AI:** {a}")
        st.markdown("---")
