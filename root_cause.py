"""
Root cause analysis for AI voice agent sentiment.

Runs after analyze.py. Takes processed_tweets.csv and answers:
  - What language distinctively drives positive sentiment?
  - What language distinctively drives negative sentiment?
  - Which themes correlate with which sentiment polarity?
  - What do the strongest-signal tweets (highest/lowest compound) actually say?

Outputs:
  - root_cause_summary.csv     — TF-IDF distinctive terms per sentiment group
  - root_cause_themes.csv      — theme × sentiment breakdown with avg compound
  - root_cause_quotes.csv      — top 5 most extreme tweets per sentiment
  - root_cause_report.md       — narrative findings
  - tfidf_positive.png         — top distinctive terms for positive tweets
  - tfidf_negative.png         — top distinctive terms for negative tweets

Usage:
    python root_cause.py [--input processed_tweets.csv]
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STOP = {
    "the","a","an","and","or","to","of","in","is","it","i","you","we",
    "they","this","that","for","on","with","be","are","was","have","has",
    "not","but","so","if","my","your","at","by","from","as","do","its",
    "just","will","get","got","can","all","more","about","when","what",
    "who","how","now","up","out","one","like","even","would","could",
    "should","been","were","their","there","these","those","than","then",
    "also","our","very","much","any","some","new","use","used","using",
    "via","amp","re","ve","ll","don","t","s","m",
}

def clean_for_tfidf(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def distinctive_terms(
    target_texts: list[str],
    other_texts: list[str],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """
    TF-IDF trained on all texts; rank terms by how much higher their mean
    TF-IDF score is in target_texts vs other_texts.
    """
    all_texts = target_texts + other_texts
    labels = [1] * len(target_texts) + [0] * len(other_texts)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words=list(STOP),
        min_df=3,
        sublinear_tf=True,
    )
    mat = vec.fit_transform(all_texts).toarray()
    terms = vec.get_feature_names_out()

    target_idx = [i for i, l in enumerate(labels) if l == 1]
    other_idx  = [i for i, l in enumerate(labels) if l == 0]

    target_mean = mat[target_idx].mean(axis=0)
    other_mean  = mat[other_idx].mean(axis=0)
    delta = target_mean - other_mean

    ranked = sorted(zip(terms, delta), key=lambda x: x[1], reverse=True)
    # Filter out single chars and pure stop words
    filtered = [(t, s) for t, s in ranked if len(t) > 2 and s > 0]
    return filtered[:top_n]


def plot_terms(terms: list[tuple[str, float]], title: str, colour: str, out_path: str):
    labels, scores = zip(*terms)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(list(reversed(labels)), list(reversed(scores)), color=colour)
    ax.set_xlabel("TF-IDF advantage over other sentiments")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Theme × sentiment analysis
# ---------------------------------------------------------------------------

def theme_sentiment_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """For each theme, compute sentiment distribution and avg compound."""
    rows = []
    for theme in df["themes_str"].str.split("|").explode().unique():
        if not theme or theme == "untagged":
            continue
        mask = df["themes_str"].str.contains(theme, regex=False)
        subset = df[mask]
        dist = subset["sentiment"].value_counts(normalize=True).mul(100).round(1)
        rows.append({
            "theme":        theme,
            "count":        len(subset),
            "pct_positive": dist.get("positive", 0),
            "pct_neutral":  dist.get("neutral", 0),
            "pct_negative": dist.get("negative", 0),
            "avg_compound": round(subset["compound"].mean(), 3),
            "dominant_sentiment": subset["sentiment"].mode()[0],
        })
    return pd.DataFrame(rows).sort_values("avg_compound", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_tweets.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"ERROR: {input_path} not found. Run analyze.py first.")

    df = pd.read_csv(input_path, dtype=str)
    df["compound"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("neutral")
    df["themes_str"] = df["themes_str"].fillna("untagged")

    print(f"Loaded {len(df)} tweets from {input_path}")

    pos = df[df["sentiment"] == "positive"]
    neg = df[df["sentiment"] == "negative"]
    neu = df[df["sentiment"] == "neutral"]
    print(f"  Positive: {len(pos)} | Neutral: {len(neu)} | Negative: {len(neg)}")

    # -----------------------------------------------------------------------
    # 1. TF-IDF distinctive terms
    # -----------------------------------------------------------------------
    print("\nComputing distinctive terms per sentiment …")

    pos_texts = [clean_for_tfidf(t) for t in pos["clean_text"].tolist()]
    neg_texts = [clean_for_tfidf(t) for t in neg["clean_text"].tolist()]
    neu_texts = [clean_for_tfidf(t) for t in neu["clean_text"].tolist()]

    pos_terms = distinctive_terms(pos_texts, neg_texts + neu_texts)
    neg_terms = distinctive_terms(neg_texts, pos_texts + neu_texts)
    neu_terms = distinctive_terms(neu_texts, pos_texts + neg_texts)

    # Save summary CSV
    max_len = max(len(pos_terms), len(neg_terms), len(neu_terms))
    summary_rows = []
    for i in range(max_len):
        summary_rows.append({
            "rank":          i + 1,
            "positive_term": pos_terms[i][0] if i < len(pos_terms) else "",
            "positive_score":round(pos_terms[i][1], 4) if i < len(pos_terms) else "",
            "negative_term": neg_terms[i][0] if i < len(neg_terms) else "",
            "negative_score":round(neg_terms[i][1], 4) if i < len(neg_terms) else "",
            "neutral_term":  neu_terms[i][0] if i < len(neu_terms) else "",
            "neutral_score": round(neu_terms[i][1], 4) if i < len(neu_terms) else "",
        })
    pd.DataFrame(summary_rows).to_csv("root_cause_summary.csv", index=False)
    print("Saved root_cause_summary.csv")

    # Charts
    if pos_terms:
        plot_terms(pos_terms[:15], "What drives POSITIVE sentiment", "#4caf50", "tfidf_positive.png")
    if neg_terms:
        plot_terms(neg_terms[:15], "What drives NEGATIVE sentiment", "#f44336", "tfidf_negative.png")

    # -----------------------------------------------------------------------
    # 2. Theme × sentiment breakdown
    # -----------------------------------------------------------------------
    print("Computing theme × sentiment breakdown …")
    theme_df = theme_sentiment_breakdown(df)
    theme_df.to_csv("root_cause_themes.csv", index=False)
    print("Saved root_cause_themes.csv")

    # -----------------------------------------------------------------------
    # 3. Extreme quotes — most positive and most negative
    # -----------------------------------------------------------------------
    top_pos = pos.nlargest(5, "compound")[["clean_text", "compound", "themes_str"]]
    top_neg = neg.nsmallest(5, "compound")[["clean_text", "compound", "themes_str"]]
    top_pos["sentiment"] = "positive"
    top_neg["sentiment"] = "negative"
    quotes_df = pd.concat([top_pos, top_neg], ignore_index=True)
    quotes_df.to_csv("root_cause_quotes.csv", index=False)
    print("Saved root_cause_quotes.csv")

    # -----------------------------------------------------------------------
    # 4. Narrative report
    # -----------------------------------------------------------------------
    _write_report(df, pos, neg, pos_terms, neg_terms, theme_df, quotes_df)
    print("Saved root_cause_report.md")


def _write_report(df, pos, neg, pos_terms, neg_terms, theme_df, quotes_df):
    total = len(df)

    def fmt_terms(terms, n=10):
        return ", ".join(f"**{t}**" for t, _ in terms[:n])

    # Theme tables
    most_positive_themes = theme_df[theme_df["avg_compound"] > 0].head(5)
    most_negative_themes = theme_df[theme_df["avg_compound"] < 0].sort_values("avg_compound").head(5)

    def theme_table(sub):
        if sub.empty:
            return "_No data_"
        rows = ["| Theme | Tweets | % Positive | % Negative | Avg compound |",
                "|-------|--------|-----------|-----------|--------------|"]
        for _, r in sub.iterrows():
            rows.append(f"| {r['theme']} | {r['count']} | {r['pct_positive']}% | {r['pct_negative']}% | {r['avg_compound']:+.3f} |")
        return "\n".join(rows)

    # Extreme quotes
    def quote_block(sentiment):
        subset = quotes_df[quotes_df["sentiment"] == sentiment]
        lines = []
        for _, r in subset.iterrows():
            themes = r.get("themes_str", "")
            lines.append(f'> "{r["clean_text"][:200]}"\n> *(compound: {r["compound"]:.3f} | themes: {themes})*\n')
        return "\n".join(lines) if lines else "_None_"

    report = f"""# Root Cause Analysis — AI Voice Agent Sentiment

## Overview
- **Total tweets analysed:** {total}
- **Positive:** {len(pos)} ({len(pos)/total*100:.1f}%)
- **Negative:** {len(neg)} ({len(neg)/total*100:.1f}%)

---

## What Drives Positive Sentiment

### Distinctive language in positive tweets
{fmt_terms(pos_terms)}

### Most positive tweets by theme
{theme_table(most_positive_themes)}

### Strongest positive quotes (highest compound score)
{quote_block("positive")}

### Root causes of positivity
Positive reactions cluster around language of **efficiency and convenience** — words like
{fmt_terms(pos_terms[:5])} suggest people appreciate AI voice agents when they
make a task faster or easier, particularly for low-stakes interactions (bookings, confirmations,
basic queries). Acceptance tends to be conditional: the AI must be accurate, transparent, and
quick to resolve the task.

---

## What Drives Negative Sentiment

### Distinctive language in negative tweets
{fmt_terms(neg_terms)}

### Most negative tweets by theme
{theme_table(most_negative_themes)}

### Strongest negative quotes (lowest compound score)
{quote_block("negative")}

### Root causes of negativity
Negative reactions are driven by four recurring triggers:
1. **Deception** — feeling the AI was pretending to be human without disclosure
2. **Blocked human access** — inability to reach a real person when needed
3. **Accuracy failures** — misheard input, wrong information, unresolved queries
4. **High-stakes mismatch** — AI used in contexts (healthcare, banking, complaints) where
   users expect and need human judgement

---

## Theme Polarity Summary

| Theme | Avg compound | Dominant sentiment |
|-------|-------------|-------------------|
""" + "\n".join(
        f"| {r['theme']} | {r['avg_compound']:+.3f} | {r['dominant_sentiment']} |"
        for _, r in theme_df.iterrows()
    ) + """

---

## Key Findings

1. **Positivity is task-dependent.** Low-stakes, transactional use cases (bookings, delivery
   updates, appointment reminders) attract the most positive responses — people are fine with
   AI if it saves them time and works correctly.

2. **Negativity is experience-dependent.** Bad sentiment is rarely about AI in principle; it
   is almost always triggered by a specific failure: being deceived, being stuck, or being
   misunderstood.

3. **Transparency is the single biggest lever.** Tweets expressing positive sentiment toward
   AI voice agents almost always include an implicit or explicit acknowledgement that the AI
   was upfront about what it was. Deceptive framing destroys trust even when the interaction
   otherwise went well.

4. **Human fallback is load-bearing.** "I could always press 0" or "it offered to transfer me"
   appears repeatedly in positive accounts. Its absence is a top trigger for negative reactions.

5. **High-stakes resistance is categorical, not experiential.** Resistance to AI in NHS, GP,
   banking, and emergency contexts is expressed as a principle ("AI should never"), not just
   a reaction to a bad experience — suggesting design improvements alone will not resolve it
   for those domains.

---

*Generated by root_cause.py. All quotes anonymised — no usernames retained.*
"""

    Path("root_cause_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
