"""
app.py
Streamlit dashboard for Shein review sentiment + topic analysis.
Reads: data/sample_processed.csv (output of preprocessing.py -> sentiment.py -> topic_modeling.py)
"""

import streamlit as st
import pandas as pd

from src.visualization import sentiment_bar, get_summary_metrics, get_topic_cards

st.set_page_config(page_title="Shein Review Insights", layout="wide")

DATA_PATH = "data/sample_processed.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_text"] = df["clean_text"].fillna("")
    return df


df = load_data(DATA_PATH)
metrics = get_summary_metrics(df)

st.title("Shein Review Insights")
st.caption(f"Static dataset — {metrics['total']:,} analyzed reviews")

# ---- Top metric cards ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Volume", f"{metrics['total']:,}")
c2.metric("Positive", f"{metrics['positive']:,}", f"{metrics['positive_pct']}%")
c3.metric("Negative", f"{metrics['negative']:,}", f"{metrics['negative_pct']}%")
c4.metric("Net Sentiment Score", f"{metrics['net_sentiment']:+.1f}")

st.plotly_chart(sentiment_bar(df), use_container_width=True)

# ---- Topic cards ----
st.subheader("Top Topics")
cards = get_topic_cards(df)

cols = st.columns(3)
for i, card in enumerate(cards):
    with cols[i % 3]:
        st.markdown(f"**{card['label']}**")
        st.markdown(f"{card['count']} reviews")
        st.markdown(f":red[{card['neg_pct']}% Neg]")
        st.divider()
