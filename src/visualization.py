"""
visualization.py
Reusable chart + card-data builders for the Streamlit dashboard (app.py).
"""

import pandas as pd
import plotly.graph_objects as go


def sentiment_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal stacked bar: positive (green) / neutral (gray) / negative (red)."""
    counts = df["sentiment"].value_counts()
    total = len(df)

    pos = counts.get("positive", 0)
    neu = counts.get("neutral", 0)
    neg = counts.get("negative", 0)

    fig = go.Figure()
    fig.add_bar(x=[pos], y=[""], orientation="h", name="Positive",
                marker_color="#22c55e",
                text=f"{pos/total*100:.1f}%", textposition="inside")
    fig.add_bar(x=[neu], y=[""], orientation="h", name="Neutral",
                marker_color="#9ca3af",
                text=f"{neu/total*100:.1f}%", textposition="inside")
    fig.add_bar(x=[neg], y=[""], orientation="h", name="Negative",
                marker_color="#ef4444",
                text=f"{neg/total*100:.1f}%", textposition="inside")

    fig.update_layout(
        barmode="stack",
        height=90,
        showlegend=True,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def get_summary_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    pos = (df["sentiment"] == "positive").sum()
    neg = (df["sentiment"] == "negative").sum()
    neu = (df["sentiment"] == "neutral").sum()

    pos_pct = pos / total * 100
    neg_pct = neg / total * 100

    return {
        "total": total,
        "positive": pos,
        "positive_pct": round(pos_pct, 1),
        "negative": neg,
        "negative_pct": round(neg_pct, 1),
        "neutral": neu,
        "net_sentiment": round(pos_pct - neg_pct, 1),
    }


def get_topic_cards(df: pd.DataFrame, top_n: int = 6) -> list[dict]:
    """Build the data each topic card on the dashboard needs, straight from the data."""
    cards = []
    grouped = df.groupby("topic_label")

    for label, group in grouped:
        if label == "Miscellaneous/Unclustered":
            continue  # skip the outlier bucket on the cards view
        n = len(group)
        neg_pct = (group["sentiment"] == "negative").mean() * 100
        cards.append({
            "label": label,
            "count": n,
            "neg_pct": round(neg_pct, 1),
        })

    cards.sort(key=lambda c: c["count"], reverse=True)
    return cards[:top_n]
