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


def star_sentiment(score: int) -> str:
    """Star rating as a ground-truth proxy: 1-2 negative, 3 neutral, 4-5 positive."""
    if score <= 2:
        return "negative"
    return "neutral" if score == 3 else "positive"


def add_star_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'sentiment_star'. The dashboard (app.py) uses this rule instead of VADER."""
    df["sentiment_star"] = df["score"].apply(star_sentiment)
    return df


def validate_against_stars(df: pd.DataFrame) -> None:
    """Compare VADER labels with star-based labels and print where they disagree."""
    df = add_star_sentiment(df.copy())
    agreement = (df["sentiment"] == df["sentiment_star"]).mean() * 100
    low = df[df["score"] <= 2]
    print(f"VADER agrees with the star rating on {agreement:.1f}% of reviews")
    print(f"1-2 star reviews labeled positive by VADER: "
          f"{(low['sentiment'] == 'positive').sum()} of {len(low)}")
    print(pd.crosstab(df["score"], df["sentiment"]))


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df = add_sentiment(df)

    counts = df["sentiment"].value_counts()
    print(counts)
    pos_pct = (counts.get("positive", 0) / len(df)) * 100
    neg_pct = (counts.get("negative", 0) / len(df)) * 100
    print(f"Positive: {pos_pct:.1f}%  Negative: {neg_pct:.1f}%  Net: {pos_pct - neg_pct:.1f}")

    if "score" in df.columns:
        validate_against_stars(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
