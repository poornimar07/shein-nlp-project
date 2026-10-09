import inspect
import inspect
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Shein Review Insights", page_icon="🟣", layout="wide")

BASE = Path(__file__).parent
# works with the CSV in data/ (recommended structure) or next to app.py
CSV_PATH = next(
    (p for p in [BASE / "data" / "sample_processed.csv", BASE / "sample_processed.csv"] if p.exists()),
    BASE / "data" / "sample_processed.csv",
)
NO_TOPIC = ["Miscellaneous/Unclustered", "Other / Unlabeled"]
RENAME = {
    "Quality & Fabric Issues": "Quality & Fabric Feedback",
    "Sizing & Fit Issues": "Sizing & Fit Feedback",
}
STOPWORDS = set("""a about after all also am an and any app are as at be because been but by can could did do does
don even every for from get got had has have he her here him his how i if in is it its ive just like me more most
much my no not now of on one only or other our out over really so some than that the their them then there these
they this to too up us very was we were what when which who will with would you your shein im one one s t
order orders item items ordered""".split())

# ---------- Theme ----------
st.markdown(
    """
<style>
  .stApp { background-color:#0b0f1a; color:#e5e7eb; }
  header[data-testid="stHeader"] { display:none; }
  .block-container { padding-top:1.5rem; max-width:1300px; }
  section[data-testid="stSidebar"] { background-color:#10141f; border-right:1px solid #1f2533; }
  h1,h2,h3,h4 { color:#ffffff !important; }
  .card { background:#131826; border:1px solid #1f2533; border-radius:14px; padding:18px; margin-bottom:14px; }
  .label { color:#8b93a7; font-size:13px; font-weight:600; }
  .value { color:#ffffff; font-size:32px; font-weight:800; margin:4px 0; }
  .sub { font-size:13px; font-weight:600; }
  .pos { color:#34d399; } .neg { color:#f87171; } .neu { color:#9ca3af; }
  .topic-name { color:#ffffff; font-size:17px; font-weight:700; margin:2px 0 6px 0; }
  .pill { display:inline-block; background:#1c2233; color:#cbd5e1; border-radius:20px; padding:4px 12px; font-size:12px; margin:0 6px 6px 0; }
  .badge-neg { background:#3a1f22; color:#f87171; border-radius:20px; padding:3px 10px; font-size:12px; font-weight:700; float:right; }
  .badge-ok { background:#173326; color:#34d399; border-radius:20px; padding:3px 10px; font-size:12px; font-weight:700; float:right; }
  .quote { color:#e5e7eb; font-style:italic; }
</style>
""",
    unsafe_allow_html=True,
)


def html(s: str) -> None:
    st.markdown(s.replace("\n", " "), unsafe_allow_html=True)


_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig) -> None:
    fig.update_layout(
        plot_bgcolor="#131826", paper_bgcolor="#131826", font_color="#e5e7eb",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(gridcolor="#1f2533")
    fig.update_yaxes(gridcolor="#1f2533")
    cfg = {"displayModeBar": False}
    if _PLOTLY_HAS_WIDTH:  # newer Streamlit
        st.plotly_chart(fig, width="stretch", config=cfg)
    else:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=cfg)


# ---------- Data ----------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["topic_label"] = df["topic_label"].fillna("Other / Unlabeled").replace(RENAME)
    # star rating is the ground truth: VADER mislabeled ~30% of 1-2 star reviews as positive
    df["sentiment"] = df["score"].apply(
        lambda s: "negative" if s <= 2 else ("neutral" if s == 3 else "positive")
    )
    df["is_complaint"] = df["score"] <= 2
    return df


def top_keywords(texts, n=12):
    words = Counter()
    for t in texts.dropna():
        for w in re.findall(r"[a-z]{3,}", t.lower()):
            if w not in STOPWORDS:
                words[w] += 1
    return [w for w, _ in words.most_common(n)]


@st.cache_data
def load_data() -> pd.DataFrame:
    return prepare(pd.read_csv(CSV_PATH))


full = load_data()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🟣 Sentix NLP")
    st.caption("SHEIN review intelligence")
    sent_filter = st.multiselect(
        "Sentiment", ["positive", "neutral", "negative"], default=["positive", "neutral", "negative"]
    )
    topic_options = sorted(t for t in full["topic_label"].unique() if t not in NO_TOPIC)
    topic_filter = st.multiselect("Topic (empty = all)", topic_options)
    st.markdown("---")
    st.caption("Sentiment is based on star rating: 1-2 negative, 3 neutral, 4-5 positive.")

df = full[full["sentiment"].isin(sent_filter)]
if topic_filter:
    df = df[df["topic_label"].isin(topic_filter)]

if df.empty:
    st.title("Shein Review Insights")
    st.warning("No reviews match these filters. Clear a filter in the sidebar.")
    st.stop()

total = len(df)
counts = df["sentiment"].value_counts()
pos, neu, neg = (int(counts.get(k, 0)) for k in ["positive", "neutral", "negative"])
pos_pct, neu_pct, neg_pct = pos / total * 100, neu / total * 100, neg / total * 100
nss = pos_pct - neg_pct
nss_class = "pos" if nss > 0 else ("neg" if nss < 0 else "neu")

# ---------- Header ----------
st.title("Shein Review Insights")
st.caption(f"{total:,} reviews shown of {len(full):,} analyzed")

# ---------- KPIs ----------
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "Total reviews", f"{total:,}", "", "neu"),
    (c2, "Positive", f"{pos:,}", f"{pos_pct:.1f}%", "pos"),
    (c3, "Neutral", f"{neu:,}", f"{neu_pct:.1f}%", "neu"),
    (c4, "Negative", f"{neg:,}", f"{neg_pct:.1f}%", "neg"),
    (c5, "Net sentiment score", f"{nss:+.1f}", "positive minus negative", nss_class),
]
for col, label, value, sub, cls in kpis:
    with col:
        html(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub {cls}">{sub}</div></div>')

# stacked sentiment bar
bar = go.Figure()
for name, pct, color in [("Positive", pos_pct, "#34d399"), ("Neutral", neu_pct, "#6b7280"), ("Negative", neg_pct, "#f87171")]:
    bar.add_trace(go.Bar(x=[pct], y=["Sentiment"], orientation="h", name=name, marker_color=color,
                         text=[f"{pct:.1f}%" if pct >= 5 else ""], textposition="inside"))
bar.update_layout(barmode="stack", height=70, showlegend=False)
bar.update_xaxes(visible=False)
bar.update_yaxes(visible=False)
show(bar)

# ---------- Charts ----------
left, right = st.columns(2)
with left:
    st.markdown("#### Sentiment distribution")
    sc = df["sentiment"].value_counts().reset_index()
    sc.columns = ["sentiment", "count"]
    pie = px.pie(sc, names="sentiment", values="count", hole=0.55, color="sentiment",
                 color_discrete_map={"positive": "#34d399", "negative": "#f87171", "neutral": "#6b7280"})
    show(pie)

complaints = df[df["is_complaint"] & ~df["topic_label"].isin(NO_TOPIC)]
with right:
    st.markdown("#### Top complaint topics (1-2 star reviews)")
    if complaints.empty:
        st.info("No labeled complaints in this selection.")
    else:
        cd = complaints["topic_label"].value_counts().head(10).reset_index()
        cd.columns = ["topic", "count"]
        cb = px.bar(cd, x="count", y="topic", orientation="h", color_discrete_sequence=["#a78bfa"])
        cb.update_layout(yaxis={"categoryorder": "total ascending"})
        show(cb)

# ---------- Topics ----------
st.markdown("## Top topics")
labeled = df[~df["topic_label"].isin(NO_TOPIC)]
unlabeled_n = total - len(labeled)
st.caption(f"{unlabeled_n:,} reviews ({unlabeled_n / total * 100:.0f}%) are BERTopic outliers or belong to small topics that were not named, so they are not shown here.")
topic_counts = labeled["topic_label"].value_counts().head(6)
cols = st.columns(3)
for i, (topic, n) in enumerate(topic_counts.items()):
    neg_rate = (labeled[(labeled["topic_label"] == topic)]["sentiment"] == "negative").mean() * 100
    badge = "badge-neg" if neg_rate > 40 else "badge-ok"
    with cols[i % 3]:
        html(f'<div class="card"><span class="{badge}">{neg_rate:.0f}% negative</span><div class="label">{n:,} reviews</div><div class="topic-name">{topic}</div></div>')

# ---------- Keywords ----------
st.markdown("## Trending keywords in negative reviews")
neg_texts = df[df["sentiment"] == "negative"]["clean_text"]
kws = top_keywords(neg_texts)
if kws:
    html("".join(f'<span class="pill">{k}</span>' for k in kws))
else:
    st.caption("No negative reviews in this selection.")

# ---------- Business insights ----------
st.markdown("## Business recommendations")
insights = []
big = labeled.groupby("topic_label")["sentiment"].agg(n="size", neg=lambda s: (s == "negative").mean() * 100)
big = big[big["n"] >= 30].sort_values("neg", ascending=False)
if not big.empty:
    t = big.index[0]
    insights.append((f"Fix {t}", f"{big.iloc[0]['neg']:.0f}% of its {int(big.iloc[0]['n'])} reviews are negative, the highest of any topic. Review the process and support response time."))
if not complaints.empty:
    t = complaints["topic_label"].value_counts().index[0]
    k = int(complaints["topic_label"].value_counts().iloc[0])
    insights.append((f"Biggest complaint source: {t}", f"{k} low-star reviews, {k / len(complaints) * 100:.0f}% of all labeled complaints."))
insights.append(("Protect what works", f"{pos_pct:.0f}% of reviews are 4-5 stars. Keep the pricing and variety customers praise."))
icols = st.columns(len(insights))
for col, (title, body) in zip(icols, insights):
    with col:
        html(f'<div class="card"><div class="topic-name">{title}</div><div class="label">{body}</div></div>')

# ---------- Recent negative reviews ----------
st.markdown("## Recent negative reviews")
neg_df = df[df["sentiment"] == "negative"]
if neg_df.empty:
    st.caption("No negative reviews in this selection.")
else:
    for _, r in neg_df.sort_values("at", ascending=False).head(5).iterrows():
        text = str(r["content"])[:180].replace("<", "&lt;").replace(">", "&gt;")
        html(f'<div class="card"><div class="quote">"{text}..."</div><div style="margin-top:8px;"><span class="pill">{r["topic_label"]}</span><span class="neg" style="font-size:12px;">{"★" * int(r["score"])}</span></div></div>')
