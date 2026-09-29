"""
sentiment.py
Runs VADER sentiment analysis on cleaned reviews.
Input : data/sample_processed.csv  (must have 'clean_text' column)
Output: adds 'sentiment' column (positive / negative / neutral)
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

IN_PATH = "../data/sample_processed.csv"
OUT_PATH = "../data/sample_processed.csv"

analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["clean_text"].apply(get_sentiment)
    return df


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df = add_sentiment(df)

    counts = df["sentiment"].value_counts()
    print(counts)
    pos_pct = (counts.get("positive", 0) / len(df)) * 100
    neg_pct = (counts.get("negative", 0) / len(df)) * 100
    print(f"Positive: {pos_pct:.1f}%  Negative: {neg_pct:.1f}%  Net: {pos_pct - neg_pct:.1f}")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
