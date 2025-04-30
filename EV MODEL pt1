# EV+ Soccer Betting Model with Email Alerts, Dashboard, and Automation

import pandas as pd
import numpy as np
import asyncio
import requests
from bs4 import BeautifulSoup
from scipy.stats import poisson
from understatapi import UnderstatAPI
import schedule
import time
from datetime import datetime
from betfairlightweight import APIClient
from betfairlightweight.filters import market_filter
import os
import streamlit as st
import smtplib
from email.message import EmailMessage

# --- Email Settings ---
EMAIL_ADDRESS = 'your_email@example.com'
EMAIL_PASSWORD = 'your_email_password'
ALERT_RECIPIENT = 'recipient@example.com'

# --- Betfair Credentials ---
USERNAME = 'your_username'
PASSWORD = 'your_password'
APP_KEY = 'your_app_key'
CERTS_PATH = '/path/to/certs/'

# --- 1. Load Team Stats from Understat ---
async def load_team_stats():
    understat = UnderstatAPI()
    barca_data = await understat.get_team_stats("Barcelona", 2024)
    inter_data = await understat.get_team_stats("Inter", 2024)

    data = pd.DataFrame({
        'team': ['Barcelona', 'Inter Milan'],
        'xG_for': [barca_data['xG'], inter_data['xG']],
        'xG_against': [barca_data['xGA'], inter_data['xGA']],
        'recent_form_weight': [1.05, 0.95]
    })
    return data

# --- 2. Predict Match Score Probabilities Using Poisson ---
def poisson_pred(xG_for_home, xGA_away, xG_for_away, xGA_home):
    home_goals_avg = (xG_for_home + xGA_away) / 2
    away_goals_avg = (xG_for_away + xGA_home) / 2

    max_goals = 5
    probs = np.zeros((max_goals + 1, max_goals + 1))

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i][j] = poisson.pmf(i, home_goals_avg) * poisson.pmf(j, away_goals_avg)

    return probs

# --- 3. Calculate Market Value / Edge ---
def calc_edge(prob_win, market_odds):
    fair_odds = 1 / prob_win
    edge = (market_odds - fair_odds) / fair_odds
    return edge

# --- 4. Kelly Criterion ---
def kelly_fraction(prob_win, odds, kelly_scale=0.25):
    b = odds - 1
    q = 1 - prob_win
    kelly = ((b * prob_win - q) / b) * kelly_scale
    return max(kelly, 0)

# --- 5. Scrape FBref for Additional Stats ---
def scrape_fbref_team_stats(team_url):
    response = requests.get(team_url)
    tables = pd.read_html(response.text)
    return tables[0] if tables else None

# --- 6. Fetch Betfair Odds ---
def fetch_betfair_odds():
    client = APIClient(USERNAME, PASSWORD, app_key=APP_KEY, certs=CERTS_PATH)
    client.login()
    catalogue = client.betting.list_market_catalogue(
        filter=market_filter(event_type_ids=['1'], market_countries=['ES'], market_type_codes=['MATCH_ODDS']),
        max_results=5
    )
    odds_data = []
    for market in catalogue:
        odds_data.append({"market_name": market.market_name, "market_id": market.market_id})
        print(f"Market: {market.market_name}, ID: {market.market_id}")
    return odds_data

# --- 7. Send Email Alert ---
def send_email_alert(subject, content):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ALERT_RECIPIENT
    msg.set_content(content)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

# --- 8. Daily Update Function ---
def daily_model_run():
    print(f"Running model: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    loop = asyncio.get_event_loop()
    stats = loop.run_until_complete(load_team_stats())

    barca = stats[stats['team'] == 'Barcelona'].iloc[0]
    inter = stats[stats['team'] == 'Inter Milan'].iloc[0]

    probs_matrix = poisson_pred(barca['xG_for'], inter['xG_against'], inter['xG_for'], barca['xG_against'])
    home_win_prob = np.sum(np.tril(probs_matrix, -1))
    draw_prob = np.sum(np.diag(probs_matrix))
    away_win_prob = np.sum(np.triu(probs_matrix, 1))

    market_odds_home = 2.00  # placeholder, should be replaced by Betfair odds
    edge = calc_edge(home_win_prob, market_odds_home)
    stake_fraction = kelly_fraction(home_win_prob, market_odds_home)

    print(f"Home Win Prob: {home_win_prob:.2%}, Draw: {draw_prob:.2%}, Away Win: {away_win_prob:.2%}")
    print(f"Edge vs Market: {edge:.2%}, Recommended Stake (Kelly 0.25): {stake_fraction:.2%}")

    barca_fbref_url = 'https://fbref.com/en/squads/206d90db/Barcelona-Stats'
    fbref_data = scrape_fbref_team_stats(barca_fbref_url)
    if fbref_data is not None:
        print("Sample FBref Stats:")
        print(fbref_data.head())

    odds_info = fetch_betfair_odds()

    # --- 9. Log Daily Results to CSV ---
    results = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "market_odds_home": market_odds_home,
        "edge": edge,
        "kelly_stake": stake_fraction
    }])
    log_path = "daily_betting_log.csv"
    if os.path.exists(log_path):
        results.to_csv(log_path, mode='a', header=False, index=False)
    else:
        results.to_csv(log_path, index=False)

    # --- 10. Trigger Email Alert on Strong Edge ---
    if edge > 0.15:
        content = f"High value bet detected!\nHome Win Prob: {home_win_prob:.2%}\nEdge: {edge:.2%}\nStake: {stake_fraction:.2%}"
        send_email_alert("High EV Bet Alert", content)

# --- 11. Streamlit Dashboard ---
def dashboard():
    st.title("EV+ Betting Dashboard")
    st.subheader("Latest Predictions")
    if os.path.exists("daily_betting_log.csv"):
        df = pd.read_csv("daily_betting_log.csv")
        st.dataframe(df.tail(5))
        st.line_chart(df.set_index("date")["edge"])
    else:
        st.write("No data available yet. Please run the model first.")

# --- 12. Schedule Daily Execution ---
schedule.every().day.at("07:00").do(daily_model_run)  # Adjust time as needed

if __name__ == "__main__":
    dashboard()  # To run dashboard as main app
    print("Starting scheduler... Press CTRL+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)
