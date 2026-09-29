"""
preprocessing.py
Cleans raw Shein review text before sentiment/topic modeling.
Input : data/Shein.csv           (raw scraped reviews)
Output: data/sample_processed.csv (cleaned + ready for sentiment.py)
"""

import re
import pandas as pd

RAW_PATH = "../data/Shein.csv"
OUT_PATH = "../data/sample_processed.csv"


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emojis/punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"[^a-z\s]", " ", text)                   # keep letters only
    text = re.sub(r"\s+", " ", text).strip()                # collapse spaces
    return text


def load_and_clean(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # drop exact duplicate reviews and empty content
    df = df.drop_duplicates(subset="content")
    df = df.dropna(subset=["content"])

    df["clean_text"] = df["content"].apply(clean_text)

    # drop rows that became empty after cleaning
    df = df[df["clean_text"].str.strip() != ""]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Rows after cleaning: {len(df)}")
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
