"""
topic_modeling.py
Runs BERTopic on cleaned reviews to discover topics/complaint clusters.
Input : data/sample_processed.csv (must have 'clean_text' column)
Output: adds 'topic_id' and 'topic_label' columns
"""

import pandas as pd
from bertopic import BERTopic

IN_PATH = "../data/sample_processed.csv"
OUT_PATH = "../data/sample_processed.csv"

# Manually mapped labels for the top topics found by BERTopic.
# -1 is always the "outlier / could not cluster" bucket.
TOPIC_LABELS = {
    -1: "Miscellaneous/Unclustered",
    0: "General Positive - Shopping Experience",
    1: "General Positive - App Love",
    2: "Refund & Order Complaints",
    3: "Shipping - Positive",
    4: "Quality & Fabric Issues",
    5: "Sizing & Fit Issues",
    6: "Sizing - Plus Size",
    7: "Shipping - Fast Delivery Praise",
    8: "General Positive - Best App",
    9: "Customer Service Complaints",
    10: "App Navigation/Usability",
    11: "Rewards/Points Program",
    12: "Refund & Missing Package Complaints",
    13: "Delivery Time Issues",
    14: "General Positive - Recommend",
    15: "In-App Games/Scam Complaints",
    16: "Ease of Use - Positive",
    17: "Delivery/Shipping Weeks",
    18: "General Positive - Great Place",
}


def run_topic_model(df: pd.DataFrame):
    docs = df["clean_text"].fillna("").tolist()

    topic_model = BERTopic(min_topic_size=15, verbose=True)
    topics, _ = topic_model.fit_transform(docs)

    df["topic_id"] = topics
    df["topic_label"] = df["topic_id"].map(TOPIC_LABELS).fillna("Unlabeled")

    return df, topic_model


def get_topic_keywords(topic_model, topic_id: int, top_n: int = 5):
    """Pull real keywords for a topic straight from the fitted model."""
    return [word for word, _ in topic_model.get_topic(topic_id)[:top_n]]


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df, model = run_topic_model(df)

    print(df["topic_label"].value_counts())
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
