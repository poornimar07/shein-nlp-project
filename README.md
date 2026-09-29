# Shein Review Insights

Sentiment + topic analysis dashboard built on a static dataset of Shein app reviews.

## Pipeline

```
data/Shein.csv (raw reviews)
      │
      ▼
src/preprocessing.py   -> cleans text
      │
      ▼
src/sentiment.py       -> VADER sentiment (positive/negative/neutral)
      │
      ▼
src/topic_modeling.py  -> BERTopic clustering + topic labels
      │
      ▼
data/sample_processed.csv
      │
      ▼
app.py (Streamlit dashboard)
```

## Notes

- Dataset is a **static CSV**, not a live/real-time feed.
- Sentiment: VADER. Topics: BERTopic (c-TF-IDF keywords).
- Topic labels were manually assigned from each topic's top BERTopic terms.
- Topic `-1` is BERTopic's outlier/unclustered bucket, not a "real" topic.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
