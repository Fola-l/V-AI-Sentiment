# UK Twitter/X — AI Voice Agent Sentiment Analysis

A fast NLP pipeline that scrapes UK-relevant tweets about AI voice agents answering service calls, runs sentiment analysis and theme tagging, identifies root causes of positive and negative reactions, and produces a full PDF + Word report with visualisations.

---

## Research Question

> How do people in the UK perceive AI voice agents answering phone calls for services like taxis, restaurants, salons, bookings, and customer service — especially when the voice is smooth and natural?

---

## Pipeline Overview

```
scrape_tweets.py   →   tweets_raw.json + tweets.csv
analyze.py         →   processed_tweets.csv + report.md + charts
root_cause.py      →   root_cause_report.md + TF-IDF charts
report_pdf.py      →   AI_Voice_Agent_Report.pdf
report_word.py     →   AI_Voice_Agent_Report.docx
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Fola-l/V-AI-Sentiment.git
cd V-AI-Sentiment
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and set your twitterapi.io key:
# API_KEY=your_key_here
```

Get a key at [twitterapi.io](https://twitterapi.io). The search endpoint used is:
`GET https://api.twitterapi.io/twitter/tweet/advanced_search`

---

## Running the Pipeline

### Step 1 — Scrape tweets

```bash
python scrape_tweets.py --max-per-query 100
```

Runs 19 UK-focused search queries across topics like `"AI phone call"`, `"virtual receptionist"`, `"GP appointment" AI`, etc. Deduplicates across queries.

**Outputs:** `tweets_raw.json`, `tweets.csv`, `logs/raw_*.json` (one per query)

---

### Step 2 — Sentiment & theme analysis

```bash
python analyze.py --input tweets_raw.json --model vader
```

- `--model vader` — fast, no GPU needed (default)
- `--model roberta` — higher accuracy, uses `cardiffnlp/twitter-roberta-base-sentiment-latest`

**Outputs:**

| File | Description |
|------|-------------|
| `processed_tweets.csv` | All tweets with compound score, sentiment label, themes |
| `sentiment_theme_summary.csv` | Sentiment distribution per theme |
| `representative_quotes.csv` | Anonymised sample quotes per theme |
| `report.md` | Narrative markdown report |
| `sentiment_distribution.png` | Bar chart of sentiment counts |
| `theme_counts.png` | Bar chart of theme frequency |

---

### Step 3 — Root cause analysis

```bash
python root_cause.py --input processed_tweets.csv
```

Uses TF-IDF to surface the language that distinctively drives positive vs negative sentiment. Identifies theme polarity and extracts the most extreme tweets.

**Outputs:**

| File | Description |
|------|-------------|
| `root_cause_summary.csv` | Top 20 distinctive terms per sentiment group |
| `root_cause_themes.csv` | Theme x sentiment breakdown with avg compound |
| `root_cause_quotes.csv` | 5 strongest positive + 5 strongest negative tweets |
| `root_cause_report.md` | Narrative root cause findings |
| `tfidf_positive.png` | Language driving positive sentiment |
| `tfidf_negative.png` | Language driving negative sentiment |

---

### Step 4 — Generate reports

```bash
python report_pdf.py    # → AI_Voice_Agent_Report.pdf
python report_word.py   # → AI_Voice_Agent_Report.docx
```

Both are 9-page structured reports covering sentiment distribution, theme analysis, language drivers, representative quotes, research Q&A, and design recommendations. The Word doc is fully editable.

**Additional charts produced in `charts/`:**

| Chart | Description |
|-------|-------------|
| `sentiment_pie.png` | Pie chart of sentiment split |
| `compound_histogram.png` | Distribution of VADER compound scores |
| `theme_sentiment_stacked.png` | Stacked bar: sentiment split per theme |
| `theme_heatmap.png` | Heatmap of theme x sentiment percentages |
| `avg_compound_theme.png` | Average compound score per theme |
| `timeline.png` | Tweet volume and sentiment over time |

---

## Themes

Tweets are tagged with keyword rules across 10 categories:

| Theme | Keywords include |
|-------|-----------------|
| `convenience` | convenient, faster, quick, easy, efficient, saves time |
| `creepy` | creepy, weird, unsettling, scary, uncanny |
| `deceptive` | fake human, pretending, tricked, fooled, not real |
| `frustrating` | annoying, frustrating, hate, useless, terrible |
| `human_fallback` | real person, speak to someone, operator, press 0 |
| `trust_accuracy` | trust, mistake, wrong, error, misheard |
| `privacy` | privacy, data, recording, surveillance, GDPR |
| `accent_issue` | accent, Scottish, Welsh, Geordie, regional |
| `low_stakes_accept` | taxi, restaurant, booking, appointment, delivery |
| `high_stakes_resist` | NHS, GP, doctor, bank, emergency, complaint |

---

## Key Findings (387 clean tweets)

- **47.5% positive · 32% negative · 20.4% neutral**
- Acceptance is highest for low-stakes bookings when the AI is **transparent**, **accurate**, and offers a **fast human fallback**
- Negative sentiment is driven by **deception** (AI not disclosing what it is), **blocked human access**, and **accuracy failures**
- Resistance in NHS/GP/banking contexts is **categorical** — not just experiential
- A natural-sounding voice is seen as **both helpful and creepy** depending on context and disclosure

---

## Privacy

No usernames or personally identifiable information are retained in any output file. All quote samples are anonymised at the extraction stage.

---

## Requirements

```
requests
python-dotenv
pandas
vaderSentiment
transformers
torch
scikit-learn
matplotlib
seaborn
tqdm
fpdf2
python-docx
```
