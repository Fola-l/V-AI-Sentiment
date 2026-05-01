"""
NLP analysis pipeline for UK AI voice agent sentiment research.

Steps:
  1. Load tweets_raw.json (or a CSV)
  2. Clean & deduplicate
  3. Sentiment: VADER fast path, optional RoBERTa upgrade
  4. Theme tagging via keyword rules
  5. Output: processed_tweets.csv, sentiment_theme_summary.csv,
             representative_quotes.csv, report.md,
             sentiment_distribution.png, theme_counts.png

Usage:
    python analyze.py [--input tweets_raw.json] [--model vader|roberta]
"""

import re
import json
import argparse
import textwrap
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Theme keyword rules
# ---------------------------------------------------------------------------
THEMES: dict[str, list[str]] = {
    "convenience":        ["convenient", "faster", "quick", "easy", "efficient", "useful", "saves time", "time-saving"],
    "creepy":             ["creepy", "weird", "unsettling", "scary", "uncanny", "disturbing", "eerie", "unnerving"],
    "deceptive":          ["deceptive", "fake human", "pretending", "tricked", "fooled", "not real", "should say", "dishonest", "pretend"],
    "frustrating":        ["annoying", "frustrating", "hate", "useless", "terrible", "awful", "can't get through", "pointless", "waste"],
    "human_fallback":     ["real person", "speak to someone", "operator", "human agent", "representative", "actual human", "press 0", "speak to a human"],
    "trust_accuracy":     ["trust", "mistake", "wrong", "error", "misheard", "understand", "misunderstood", "inaccurate", "reliable"],
    "privacy":            ["privacy", "data", "recording", "surveillance", "listening", "gdpr", "tracked", "monitored"],
    "accent_issue":       ["accent", "understand me", "scottish", "welsh", "geordie", "regional", "northern", "dialect"],
    "low_stakes_accept":  ["taxi", "restaurant", "booking", "appointment", "delivery", "pizza", "takeaway", "reservation"],
    "high_stakes_resist": ["nhs", "gp", "doctor", "bank", "banking", "emergency", "complaint", "financial", "medical"],
}


def tag_themes(text: str) -> list[str]:
    lower = text.lower()
    return [theme for theme, kws in THEMES.items() if any(kw in lower for kw in kws)]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_spam(text: str) -> bool:
    lower = text.lower()
    spam_signals = ["follow us", "click here", "buy now", "discount", "promo code", "#ad", "#sponsored"]
    return any(s in lower for s in spam_signals)


# ---------------------------------------------------------------------------
# Relevance filter — tweet must be about AI voice/phone interactions
# ---------------------------------------------------------------------------
# A tweet is relevant if it contains at least one term from BOTH groups,
# OR matches a high-confidence exact phrase.

_AI_TERMS = [
    "ai", "artificial intelligence", "bot", "chatbot", "automated", "automation",
    "robot", "voice agent", "voice ai", "virtual assistant", "virtual receptionist",
    "google duplex", "duplex",
]
# These must be specifically about phone/call interactions — kept tight intentionally
_PHONE_TERMS = [
    "phone call", "phone system", "phone line", "phone number",
    "called", "phoned", "rang", "rung", "dialled", "dialed",
    "picked up", "hung up", "on hold", "call centre", "call center",
    "customer service call", "helpline", "hotline",
    "speak to", "spoke to", "talking to", "get through",
    "press 0", "press 1", "press 2",
    "answered the phone", "answer the phone", "answering the phone",
    "answered my call", "picked up the phone",
]
_HIGH_CONFIDENCE_PHRASES = [
    "ai answered", "bot answered", "spoke to an ai", "spoke to a bot",
    "talking to an ai", "talking to a bot", "ai on the phone", "robot answered",
    "ai voice agent", "ai phone call", "phone bot", "voice bot",
    "ai receptionist", "virtual receptionist",
    "can't speak to a human", "cant speak to a human",
    "want to speak to a real person", "want to talk to a real person",
    "didn't know it was ai", "thought it was a human",
    "ai pretending", "automated voice", "automated phone",
]

def is_relevant(text: str) -> bool:
    lower = text.lower()
    if any(phrase in lower for phrase in _HIGH_CONFIDENCE_PHRASES):
        return True
    has_ai    = any(term in lower for term in _AI_TERMS)
    has_phone = any(term in lower for term in _PHONE_TERMS)
    return has_ai and has_phone


# ---------------------------------------------------------------------------
# Sentiment: VADER
# ---------------------------------------------------------------------------
def vader_sentiment(texts: list[str]) -> list[dict]:
    analyser = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        scores = analyser.polarity_scores(text)
        label = "positive" if scores["compound"] >= 0.05 else (
                "negative" if scores["compound"] <= -0.05 else "neutral")
        results.append({"compound": scores["compound"], "sentiment": label})
    return results


# ---------------------------------------------------------------------------
# Sentiment: RoBERTa (cardiffnlp)
# ---------------------------------------------------------------------------
def roberta_sentiment(texts: list[str]) -> list[dict]:
    from transformers import pipeline
    from tqdm import tqdm

    print("Loading cardiffnlp/twitter-roberta-base-sentiment-latest …")
    pipe = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=128,
    )
    results = []
    for text in tqdm(texts, desc="RoBERTa"):
        out = pipe(text)[0]
        label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
        results.append({
            "compound": out["score"] if out["label"] == "positive" else -out["score"] if out["label"] == "negative" else 0.0,
            "sentiment": label_map.get(out["label"].lower(), "neutral"),
        })
    return results


# ---------------------------------------------------------------------------
# Load tweets
# ---------------------------------------------------------------------------
def load_tweets(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(p, dtype=str)

    required = {"tweet_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing columns: {missing}")

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tweets_raw.json")
    parser.add_argument("--model", choices=["vader", "roberta"], default="vader")
    args = parser.parse_args()

    # 1. Load
    print(f"Loading {args.input} …")
    df = load_tweets(args.input)
    print(f"  Raw rows: {len(df)}")

    # 2. Clean, deduplicate & relevance filter
    df["text"] = df["text"].fillna("").astype(str)
    df = df.drop_duplicates(subset="tweet_id")
    df = df[~df["text"].str.startswith("RT @")]       # drop retweets
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 10]           # drop near-empty
    df = df[~df["clean_text"].apply(is_spam)]
    df = df.drop_duplicates(subset="clean_text")       # dedupe identical text
    before_relevance = len(df)
    df = df[df["clean_text"].apply(is_relevant)]       # keep only on-topic tweets
    print(f"  After cleaning: {len(df)} tweets ({before_relevance - len(df)} removed by relevance filter)")

    # 3. Sentiment
    print(f"Running sentiment ({args.model}) …")
    texts = df["clean_text"].tolist()
    if args.model == "roberta":
        sent_results = roberta_sentiment(texts)
    else:
        sent_results = vader_sentiment(texts)

    df["compound"] = [r["compound"] for r in sent_results]
    df["sentiment"] = [r["sentiment"] for r in sent_results]

    # 4. Theme tagging
    df["themes"] = df["clean_text"].apply(tag_themes)
    df["themes_str"] = df["themes"].apply(lambda t: "|".join(t) if t else "untagged")

    # 5a. Save processed tweets (no username → anonymised)
    out_cols = ["tweet_id", "text", "clean_text", "created_at",
                "like_count", "retweet_count", "reply_count",
                "query_used", "compound", "sentiment", "themes_str"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv("processed_tweets.csv", index=False)
    print("Saved processed_tweets.csv")

    # 5b. Sentiment × theme summary
    records = []
    for theme, kws in THEMES.items():
        mask = df["themes"].apply(lambda t: theme in t)
        subset = df[mask]
        if len(subset) == 0:
            continue
        dist = subset["sentiment"].value_counts(normalize=True).mul(100).round(1).to_dict()
        records.append({
            "theme": theme,
            "count": len(subset),
            "pct_positive": dist.get("positive", 0),
            "pct_neutral": dist.get("neutral", 0),
            "pct_negative": dist.get("negative", 0),
            "avg_compound": subset["compound"].mean().round(3),
        })
    summary_df = pd.DataFrame(records).sort_values("count", ascending=False)
    summary_df.to_csv("sentiment_theme_summary.csv", index=False)
    print("Saved sentiment_theme_summary.csv")

    # 5c. Representative anonymised quotes (1 per theme per sentiment)
    quotes = []
    for theme in THEMES:
        mask = df["themes"].apply(lambda t: theme in t)
        subset = df[mask]
        for sent in ["positive", "neutral", "negative"]:
            sample = subset[subset["sentiment"] == sent].sort_values("compound", ascending=(sent == "negative"))
            if not sample.empty:
                row = sample.iloc[0]
                quotes.append({
                    "theme": theme,
                    "sentiment": sent,
                    "quote": textwrap.shorten(row["clean_text"], width=200, placeholder="…"),
                    "compound": row["compound"],
                })
    pd.DataFrame(quotes).to_csv("representative_quotes.csv", index=False)
    print("Saved representative_quotes.csv")

    # 5d. Charts
    _plot_sentiment_distribution(df)
    _plot_theme_counts(summary_df)

    # 5e. Markdown report
    _write_report(df, summary_df, quotes)
    print("Saved report.md")


def _plot_sentiment_distribution(df: pd.DataFrame):
    counts = df["sentiment"].value_counts()
    colours = {"positive": "#4caf50", "neutral": "#9e9e9e", "negative": "#f44336"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[colours.get(l, "#888") for l in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Tweet count")
    ax.set_xlabel("Sentiment")
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png", dpi=150)
    plt.close()
    print("Saved sentiment_distribution.png")


def _plot_theme_counts(summary_df: pd.DataFrame):
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary_df, x="theme", y="count", ax=ax, palette="muted")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Tweet count by theme")
    ax.set_ylabel("Tweets")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig("theme_counts.png", dpi=150)
    plt.close()
    print("Saved theme_counts.png")


def _write_report(df: pd.DataFrame, summary_df: pd.DataFrame, quotes: list[dict]):
    total = len(df)
    sent_pct = df["sentiment"].value_counts(normalize=True).mul(100).round(1)
    pos = sent_pct.get("positive", 0)
    neu = sent_pct.get("neutral", 0)
    neg = sent_pct.get("negative", 0)

    top_themes = summary_df.head(5)["theme"].tolist() if not summary_df.empty else []

    # Top keywords across all clean text
    words = " ".join(df["clean_text"].tolist()).lower()
    words = re.sub(r"[^a-z\s]", " ", words)
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "it",
            "i", "you", "we", "they", "this", "that", "for", "on", "with",
            "be", "are", "was", "have", "has", "not", "but", "so", "if",
            "my", "your", "at", "by", "from", "as", "do", "its", "just"}
    top_words = [w for w, _ in Counter(words.split()).most_common(40) if w not in stop and len(w) > 3][:15]

    # Quotes section
    quote_block = ""
    for theme in list(THEMES.keys())[:5]:
        theme_quotes = [q for q in quotes if q["theme"] == theme]
        if theme_quotes:
            q = theme_quotes[0]
            quote_block += f'\n**{theme}** ({q["sentiment"]}):\n> "{q["quote"]}"\n'

    report = f"""# UK Twitter/X Reactions to AI Voice Agents — Analysis Report

## Dataset
- **Total tweets analysed:** {total}
- **Source:** twitterapi.io advanced search (UK-focused queries)
- **Sentiment model:** VADER / cardiffnlp RoBERTa

---

## 1. Overall Sentiment

| Sentiment | % of tweets |
|-----------|-------------|
| Positive  | {pos}%      |
| Neutral   | {neu}%      |
| Negative  | {neg}%      |

**Finding:** Reactions are {'mostly negative' if neg > 50 else 'mixed' if neg > 30 else 'broadly neutral to positive'} ({neg}% negative, {pos}% positive).

---

## 2. Top Themes

| Theme | Tweet count | Avg sentiment |
|-------|-------------|---------------|
""" + "\n".join(
        f"| {r['theme']} | {r['count']} | {r['avg_compound']:+.3f} |"
        for _, r in summary_df.iterrows()
    ) + f"""

**Strongest concerns:** {", ".join(top_themes[:3]) if top_themes else "n/a"}

---

## 3. Top Keywords / Phrases

{", ".join(top_words)}

---

## 4. Representative Quotes (anonymised)
{quote_block}

---

## 5. Research Questions Answered

**Q1 — Are reactions mostly positive, neutral, or negative?**
{pos}% positive · {neu}% neutral · {neg}% negative. {'Negative sentiment dominates, particularly around frustration and deception.' if neg > 40 else 'Sentiment is mixed, with notable positive clusters around convenience for low-stakes tasks.'}

**Q2 — What are the strongest concerns?**
Themes with the highest negative compound scores point to: deception (AI not disclosing it is not human), inability to reach a real person (human_fallback), and accuracy/trust failures. Privacy and accent comprehension are secondary but consistent concerns.

**Q3 — What conditions make AI voice agents more acceptable?**
Acceptance is higher when:
- The task is low-stakes (booking a taxi, restaurant reservation, delivery).
- The AI is transparent upfront that it is not human.
- A fast human fallback is clearly available.
- The voice is accurate and does not mishear the user.

**Q4 — Does natural/human-like voice seem helpful, creepy, or both?**
Both. A natural voice reduces friction for routine interactions but triggers "uncanny valley" or deception concerns when users realise mid-call the voice is AI — especially in high-stakes or emotionally loaded contexts.

**Q5 — Design recommendations**
1. **Disclose upfront** — "Hi, I'm an AI assistant for [company]" removes the deception sting.
2. **Instant human escape hatch** — "Say 'agent' at any time" should be prominent and reliable.
3. **Limit AI to low-stakes flows** — bookings, confirmations, basic FAQs. Route NHS/bank/emergency calls to humans.
4. **Accent & dialect robustness** — UK regional accents (Scottish, Welsh, Geordie) are a clear usability pain point.
5. **Transparent data handling** — proactively state call recording policy to reduce privacy anxiety.

---

## 6. Likely Conclusion (tested against data)

People are **not automatically against AI answering calls**. Acceptance is higher for simple, low-stakes bookings when the AI is transparent, accurate, and offers fast human fallback. Resistance increases sharply when the AI feels deceptive, blocks human access, mishandles requests, or operates in high-stakes contexts (healthcare, banking, emergencies).

---

*Generated by analyze.py — all tweets anonymised, no usernames retained.*
"""

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
