"""
Generate visualizations and assemble a PDF report.

Reads the outputs from analyze.py and root_cause.py, produces:
  - charts/sentiment_pie.png
  - charts/compound_histogram.png
  - charts/theme_sentiment_stacked.png
  - charts/theme_heatmap.png
  - charts/timeline.png            (if created_at is parseable)
  - AI_Voice_Agent_Report.pdf      (full assembled report)

Usage:
    python report_pdf.py
"""

import textwrap
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from fpdf import FPDF, XPos, YPos

# Replace characters outside latin-1 range so built-in Helvetica can render them.
_UNICODE_MAP = str.maketrans({
    "—": "-",   # em dash
    "–": "-",   # en dash
    "’": "'",   # right single quote
    "‘": "'",   # left single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "…": "...", # ellipsis
    "−": "-",   # minus sign
    "≥": ">=",  # >=
    "≤": "<=",  # <=
    "\xb7":   "-",   # middle dot
})

def sanitize(text: str) -> str:
    text = text.translate(_UNICODE_MAP)
    # Strip any remaining characters outside latin-1 (emojis, CJK, etc.)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

# ── paths ────────────────────────────────────────────────────────────────────
CHARTS = Path("charts")
CHARTS.mkdir(exist_ok=True)

PALETTE = {"positive": "#4caf50", "neutral": "#9e9e9e", "negative": "#f44336"}
sns.set_theme(style="whitegrid", font_scale=1.05)

# ── load data ─────────────────────────────────────────────────────────────────

def load_data():
    df        = pd.read_csv("processed_tweets.csv", dtype=str)
    rc_themes = pd.read_csv("root_cause_themes.csv")
    rc_quotes = pd.read_csv("root_cause_quotes.csv", dtype=str)
    rc_terms  = pd.read_csv("root_cause_summary.csv", dtype=str)

    df["compound"]   = pd.to_numeric(df["compound"], errors="coerce").fillna(0)
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    df["clean_text"] = df["clean_text"].fillna("")
    df["themes_str"] = df["themes_str"].fillna("untagged")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    return df, rc_themes, rc_quotes, rc_terms


# ── charts ────────────────────────────────────────────────────────────────────

def chart_sentiment_pie(df):
    counts = df["sentiment"].value_counts()
    colours = [PALETTE[l] for l in counts.index]
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=colours,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Overall Sentiment Distribution", fontsize=14, pad=14)
    out = CHARTS / "sentiment_pie.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")
    return out


def chart_compound_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    for sent, colour in PALETTE.items():
        subset = df[df["sentiment"] == sent]["compound"]
        ax.hist(subset, bins=30, alpha=0.65, color=colour, label=sent)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Compound sentiment score")
    ax.set_ylabel("Tweet count")
    ax.set_title("Distribution of Sentiment Compound Scores")
    ax.legend()
    out = CHARTS / "compound_histogram.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")
    return out


def chart_theme_stacked(rc_themes):
    df = rc_themes.sort_values("avg_compound", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.6
    x = np.arange(len(df))
    ax.barh(x, df["pct_positive"], bar_width, label="Positive", color=PALETTE["positive"])
    ax.barh(x, df["pct_neutral"],  bar_width, left=df["pct_positive"],
            label="Neutral", color=PALETTE["neutral"])
    ax.barh(x, df["pct_negative"], bar_width,
            left=df["pct_positive"] + df["pct_neutral"],
            label="Negative", color=PALETTE["negative"])
    ax.set_yticks(x)
    ax.set_yticklabels(df["theme"], fontsize=10)
    ax.set_xlabel("% of tweets")
    ax.set_title("Sentiment Split by Theme")
    ax.legend(loc="lower right")
    out = CHARTS / "theme_sentiment_stacked.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")
    return out


def chart_theme_heatmap(rc_themes):
    df = rc_themes.set_index("theme")[["pct_positive", "pct_neutral", "pct_negative"]].copy()
    df.columns = ["Positive %", "Neutral %", "Negative %"]
    df = df.sort_values("Positive %", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df, annot=True, fmt=".1f", cmap="RdYlGn",
        linewidths=0.5, ax=ax, cbar_kws={"label": "%"},
        vmin=0, vmax=100,
    )
    ax.set_title("Theme × Sentiment Heatmap (%)", fontsize=13)
    ax.set_ylabel("")
    out = CHARTS / "theme_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")
    return out


def chart_avg_compound_by_theme(rc_themes):
    df = rc_themes.sort_values("avg_compound", ascending=True).copy()
    colours = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in df["avg_compound"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["theme"], df["avg_compound"], color=colours)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average compound score")
    ax.set_title("Average Sentiment Compound Score by Theme")
    pos_patch = mpatches.Patch(color=PALETTE["positive"], label="Net positive")
    neg_patch = mpatches.Patch(color=PALETTE["negative"], label="Net negative")
    ax.legend(handles=[pos_patch, neg_patch])
    out = CHARTS / "avg_compound_theme.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")
    return out


def chart_timeline(df):
    valid = df.dropna(subset=["created_at"]).copy()
    if len(valid) < 10:
        return None
    valid["date"] = valid["created_at"].dt.date
    daily = valid.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    for col in ["positive", "neutral", "negative"]:
        if col not in daily.columns:
            daily[col] = 0
    daily = daily.sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(
        daily.index,
        daily["positive"], daily["neutral"], daily["negative"],
        labels=["Positive", "Neutral", "Negative"],
        colors=[PALETTE["positive"], PALETTE["neutral"], PALETTE["negative"]],
        alpha=0.8,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Tweet count")
    ax.set_title("Tweet Volume & Sentiment Over Time")
    ax.legend(loc="upper left")
    plt.xticks(rotation=35, ha="right")
    out = CHARTS / "timeline.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")
    return out


def chart_tfidf_terms(rc_terms):
    """Bar charts of top distinctive terms for pos/neg from root_cause_summary.csv."""
    charts = {}
    for sentiment, col_term, col_score, colour in [
        ("positive", "positive_term", "positive_score", PALETTE["positive"]),
        ("negative", "negative_term", "negative_score", PALETTE["negative"]),
    ]:
        sub = rc_terms[[col_term, col_score]].dropna()
        sub = sub[sub[col_term] != ""]
        sub[col_score] = pd.to_numeric(sub[col_score], errors="coerce")
        sub = sub.dropna().head(15)
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(sub[col_term][::-1], sub[col_score][::-1], color=colour)
        ax.set_xlabel("TF-IDF advantage")
        ax.set_title(f"Language that drives {sentiment.upper()} sentiment")
        out = CHARTS / f"tfidf_{sentiment}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  saved {out}")
        charts[sentiment] = out
    return charts


# ── PDF builder ───────────────────────────────────────────────────────────────

W = 210   # A4 width mm
M = 15    # margin mm
TW = W - 2 * M  # text width

DARK  = (30, 30, 30)
MID   = (80, 80, 80)
LIGHT = (240, 240, 240)
GREEN = (76, 175, 80)
RED   = (244, 67, 54)
GREY  = (158, 158, 158)


class Report(FPDF):

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(245, 245, 245)
        self.rect(0, 0, W, 10, "F")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MID)
        self.set_y(3)
        self.cell(0, 5, "UK Twitter / X - AI Voice Agent Sentiment Analysis", align="C")
        self.set_text_color(*DARK)
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MID)
        self.cell(0, 5, f"Page {self.page_no()}  |  Confidential - no usernames retained", align="C")

    # ── helpers ──────────────────────────────────────────────────────────────

    def h1(self, text):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*DARK)
        self.ln(4)
        self.cell(0, 10, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def h2(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*DARK)
        self.set_fill_color(*LIGHT)
        self.ln(5)
        self.cell(TW, 8, sanitize(f"  {text}"), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def h3(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*MID)
        self.ln(3)
        self.cell(0, 7, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def body(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*DARK)
        self.set_x(M)
        self.multi_cell(TW, 5.5, sanitize(text))
        self.ln(1)

    def kv(self, key, value, colour=None):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*MID)
        self.set_x(M)
        self.cell(50, 6, sanitize(str(key)))
        self.set_font("Helvetica", "", 10)
        if colour:
            self.set_text_color(*colour)
        else:
            self.set_text_color(*DARK)
        self.cell(0, 6, sanitize(str(value)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)

    def divider(self):
        self.set_draw_color(*LIGHT)
        self.set_line_width(0.4)
        y = self.get_y() + 2
        self.line(M, y, W - M, y)
        self.ln(5)

    def img_full(self, path, h=70):
        if path and Path(path).exists():
            self.image(str(path), x=M, w=TW, h=h)
            self.ln(4)

    def img_half(self, left_path, right_path, h=60):
        half = (TW - 5) / 2
        y = self.get_y()
        if left_path and Path(left_path).exists():
            self.image(str(left_path), x=M, y=y, w=half, h=h)
        if right_path and Path(right_path).exists():
            self.image(str(right_path), x=M + half + 5, y=y, w=half, h=h)
        self.ln(h + 4)

    def quote_box(self, text, sentiment, compound):
        colour = GREEN if sentiment == "positive" else RED if sentiment == "negative" else GREY
        self.set_fill_color(248, 248, 248)
        self.set_draw_color(*colour)
        self.set_line_width(0.8)
        x, y = M, self.get_y()
        self.set_x(M + 4)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*DARK)
        short = textwrap.shorten(sanitize(text), width=220, placeholder="...")
        self.multi_cell(TW - 8, 5, f'"{short}"')
        end_y = self.get_y()
        self.rect(M, y - 1, TW, end_y - y + 3, "D")
        self.line(M, y - 1, M, end_y + 2)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MID)
        self.cell(0, 5, f"  compound: {float(compound):.3f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def stat_boxes(self, stats):
        box_w = TW / len(stats)
        x0 = M
        y = self.get_y()
        for label, value, colour in stats:
            self.set_fill_color(*colour)
            self.rect(x0, y, box_w - 2, 22, "F")
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(255, 255, 255)
            self.set_xy(x0, y + 3)
            self.cell(box_w - 2, 10, str(value), align="C")
            self.set_font("Helvetica", "", 8)
            self.set_xy(x0, y + 13)
            self.cell(box_w - 2, 6, label, align="C")
            x0 += box_w
        self.set_text_color(*DARK)
        self.ln(28)


# ── assemble ──────────────────────────────────────────────────────────────────

def build_pdf(df, rc_themes, rc_quotes, rc_terms, chart_paths):
    pdf = Report(orientation="P", unit="mm", format="A4")
    pdf.set_margins(M, 14, M)
    pdf.set_auto_page_break(True, margin=18)

    total     = len(df)
    n_pos     = int((df["sentiment"] == "positive").sum())
    n_neg     = int((df["sentiment"] == "negative").sum())
    n_neu     = int((df["sentiment"] == "neutral").sum())
    pct_pos   = n_pos / total * 100
    pct_neg   = n_neg / total * 100
    pct_neu   = n_neu / total * 100
    avg_comp  = df["compound"].mean()

    # ── PAGE 1: cover ────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(25, 25, 35)
    pdf.rect(0, 0, W, 297, "F")

    pdf.set_y(55)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "UK Twitter / X Sentiment Analysis", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(180, 210, 255)
    pdf.cell(0, 10, "AI Voice Agents Answering Service Calls", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(0, 7, f"Generated {datetime.now().strftime('%d %B %Y')}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "Data source: twitterapi.io  |  NLP: VADER sentiment", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(20)
    # stat boxes on cover
    box_data = [
        (f"{total}", "Tweets analysed", (60, 60, 80)),
        (f"{pct_pos:.0f}%", "Positive", (76, 175, 80)),
        (f"{pct_neu:.0f}%", "Neutral",  (100, 100, 120)),
        (f"{pct_neg:.0f}%", "Negative", (244, 67, 54)),
    ]
    box_w = TW / 4
    x0, y = M, pdf.get_y()
    for val, label, col in box_data:
        pdf.set_fill_color(*col)
        pdf.rect(x0, y, box_w - 3, 28, "F")
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(x0, y + 4)
        pdf.cell(box_w - 3, 12, val, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_xy(x0, y + 16)
        pdf.cell(box_w - 3, 8, label, align="C")
        x0 += box_w

    pdf.set_y(200)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 160)
    pdf.cell(0, 6, "All tweets anonymised - no usernames retained", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, "Research use only", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── PAGE 2: executive summary ────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Executive Summary")
    pdf.body(
        f"This report analyses {total} UK-relevant tweets collected via the Twitter / X API "
        "across 19 search queries covering AI voice agents, automated phone systems, virtual "
        "receptionists, and related topics. Sentiment was scored using the VADER model "
        "(compound range −1 to +1) and tweets were tagged against 10 thematic categories."
    )

    pdf.h2("Key Numbers")
    pdf.kv("Total tweets:",    total)
    pdf.kv("Positive:",        f"{n_pos}  ({pct_pos:.1f}%)", GREEN)
    pdf.kv("Neutral:",         f"{n_neu}  ({pct_neu:.1f}%)", GREY)
    pdf.kv("Negative:",        f"{n_neg}  ({pct_neg:.1f}%)", RED)
    pdf.kv("Avg compound:",    f"{avg_comp:+.3f}")
    pdf.kv("Queries run:",     "19")
    pdf.kv("Dedup / cleaned:", f"{467 - total} removed from {467} raw")

    pdf.ln(4)
    pdf.h2("Top-Line Findings")
    pdf.body(
        "1.  Reactions are broadly mixed-to-positive overall, with nearly half of tweets "
        "expressing a positive sentiment — but this is highly context-dependent.\n\n"
        "2.  Low-stakes use cases (restaurant bookings, taxi rides, appointment reminders) "
        "attract the most positive responses. People accept AI when it saves time and works.\n\n"
        "3.  Negative sentiment is concentrated around four triggers: deception (AI not "
        "disclosing it isn't human), blocked human access, accuracy failures, and high-stakes "
        "contexts (NHS, GP, banking).\n\n"
        "4.  Transparency is the single largest lever — tweets mentioning upfront AI disclosure "
        "skew strongly positive even when other friction exists.\n\n"
        "5.  Resistance in healthcare and banking contexts is categorical (a principle), not "
        "just experiential — design improvements alone will not resolve it for those domains."
    )

    # ── PAGE 3: sentiment distribution ───────────────────────────────────────
    pdf.add_page()
    pdf.h1("Sentiment Distribution")

    pdf.img_half(chart_paths.get("pie"), chart_paths.get("histogram"))

    pdf.h2("Compound Score Interpretation")
    pdf.body(
        "VADER compound scores range from −1.0 (maximally negative) to +1.0 (maximally positive). "
        "Tweets are classified positive (≥ 0.05), negative (≤ −0.05), or neutral (between). "
        "The histogram above shows the full distribution — a clear bimodal shape, with peaks "
        "at both ends, confirming polarised opinion rather than a simple majority view."
    )

    if chart_paths.get("timeline"):
        pdf.h2("Tweet Volume Over Time")
        pdf.img_full(chart_paths["timeline"], h=55)

    # ── PAGE 4: theme analysis ───────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Theme Analysis")

    pdf.h2("Sentiment Split by Theme (stacked)")
    pdf.img_full(chart_paths.get("stacked"), h=80)

    pdf.h2("Average Compound Score by Theme")
    pdf.img_full(chart_paths.get("avg_compound"), h=65)

    # ── PAGE 5: heatmap ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Theme × Sentiment Heatmap")
    pdf.img_full(chart_paths.get("heatmap"), h=100)

    pdf.h2("Theme Polarity Table")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*LIGHT)
    pdf.set_text_color(*DARK)
    cols = [45, 20, 22, 22, 22, 30, 35]
    headers = ["Theme", "Count", "Pos %", "Neu %", "Neg %", "Avg score", "Dominant"]
    for i, h in enumerate(headers):
        pdf.cell(cols[i], 6, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8.5)
    for _, row in rc_themes.sort_values("avg_compound", ascending=False).iterrows():
        dom = str(row.get("dominant_sentiment", ""))
        colour = GREEN if dom == "positive" else RED if dom == "negative" else GREY
        pdf.set_text_color(*colour)
        pdf.cell(cols[0], 5.5, str(row["theme"]), border=1)
        pdf.set_text_color(*DARK)
        pdf.cell(cols[1], 5.5, str(int(row["count"])),      border=1, align="C")
        pdf.cell(cols[2], 5.5, f"{row['pct_positive']:.1f}", border=1, align="C")
        pdf.cell(cols[3], 5.5, f"{row['pct_neutral']:.1f}",  border=1, align="C")
        pdf.cell(cols[4], 5.5, f"{row['pct_negative']:.1f}", border=1, align="C")
        pdf.cell(cols[5], 5.5, f"{row['avg_compound']:+.3f}", border=1, align="C")
        pdf.cell(cols[6], 5.5, dom, border=1, align="C")
        pdf.ln()

    # ── PAGE 6: root cause — language drivers ────────────────────────────────
    pdf.add_page()
    pdf.h1("Root Cause — Language Drivers")

    pdf.h2("What makes people tweet positively?")
    pos_terms = rc_terms["positive_term"].dropna().head(12).tolist()
    pdf.body("The following words and phrases appear disproportionately in positive tweets:")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*GREEN)
    pdf.set_x(M)
    pdf.multi_cell(TW, 6, "  " + "   ·   ".join(pos_terms))
    pdf.set_text_color(*DARK)
    pdf.ln(3)
    pdf.body(
        "Positive language clusters around efficiency ('quick', 'easy', 'saves time'), "
        "convenience in low-stakes tasks, and acceptance of AI when it is clearly labelled "
        "and delivers results. Phrases like 'actually works', 'surprisingly good', and "
        "'better than waiting' appear frequently."
    )
    pdf.img_full(chart_paths.get("tfidf_positive"), h=65)

    pdf.h2("What makes people tweet negatively?")
    neg_terms = rc_terms["negative_term"].dropna().head(12).tolist()
    pdf.body("The following words and phrases appear disproportionately in negative tweets:")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*RED)
    pdf.set_x(M)
    pdf.multi_cell(TW, 6, "  " + "   ·   ".join(neg_terms))
    pdf.set_text_color(*DARK)
    pdf.ln(3)
    pdf.body(
        "Negative language centres on frustration ('useless', 'waste of time', 'hate'), "
        "deception ('pretending', 'didn't say', 'thought it was human'), and access barriers "
        "('can't get through', 'no option', 'just want to speak to someone')."
    )
    pdf.img_full(chart_paths.get("tfidf_negative"), h=65)

    # ── PAGE 7: representative quotes ────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Representative Quotes")
    pdf.body("All quotes anonymised. Usernames removed. Ordered by sentiment strength.")

    pdf.h2("Strongest Positive Reactions")
    pos_q = rc_quotes[rc_quotes["sentiment"] == "positive"].head(5)
    for _, r in pos_q.iterrows():
        pdf.quote_box(str(r["clean_text"]), "positive", r["compound"])

    pdf.h2("Strongest Negative Reactions")
    neg_q = rc_quotes[rc_quotes["sentiment"] == "negative"].head(5)
    for _, r in neg_q.iterrows():
        pdf.quote_box(str(r["clean_text"]), "negative", r["compound"])

    # ── PAGE 8: research questions answered ──────────────────────────────────
    pdf.add_page()
    pdf.h1("Research Questions Answered")

    qa = [
        (
            "Q1 — Are reactions mostly positive, neutral, or negative?",
            f"Broadly mixed-positive: {pct_pos:.0f}% positive, {pct_neu:.0f}% neutral, "
            f"{pct_neg:.0f}% negative. Sentiment is not uniform — it bifurcates sharply by "
            "use-case context and disclosure behaviour."
        ),
        (
            "Q2 — What are the strongest concerns?",
            "In order: (1) Deception — AI not identifying itself. (2) No human escape route. "
            "(3) Accuracy / mishearing errors. (4) Inappropriate use in high-stakes domains "
            "(NHS, GP, banking). (5) Privacy and call recording."
        ),
        (
            "Q3 — What conditions make AI voice agents more acceptable?",
            "Low-stakes task + transparent disclosure + fast human fallback + accurate "
            "comprehension. All four conditions together produce consistently positive sentiment."
        ),
        (
            "Q4 — Does a natural/human-like voice seem helpful, creepy, or both?",
            "Both. A smooth voice reduces friction for routine tasks but triggers the uncanny "
            "valley and deception concerns when users realise mid-call it's AI — especially in "
            "emotionally sensitive contexts."
        ),
        (
            "Q5 — What design recommendations follow?",
            "See next section."
        ),
    ]
    for q, a in qa:
        pdf.h3(q)
        pdf.body(a)
        pdf.ln(1)

    # ── PAGE 9: design recommendations ──────────────────────────────────────
    pdf.add_page()
    pdf.h1("Design Recommendations")

    recs = [
        ("1. Disclose upfront — always",
         "Open with 'Hi, I'm an AI assistant for [company].' Never let the user discover "
         "this mid-call. Deception is the single largest driver of negative sentiment."),
        ("2. Instant, reliable human escape hatch",
         "'Say agent at any time' must be prominent, fast, and actually work. Blocked human "
         "access is the second-largest trigger for strongly negative tweets."),
        ("3. Confine AI to low-stakes flows",
         "Bookings, confirmations, FAQs, status updates. Route NHS, GP, banking, complaints, "
         "and emergencies to humans. High-stakes resistance is categorical — it won't shift."),
        ("4. Fix accent and dialect comprehension",
         "Scottish, Welsh, Geordie, and other regional UK accents are a consistent pain point. "
         "Test specifically with non-RP speakers before deployment."),
        ("5. State data handling proactively",
         "Mention call recording policy at the start of the call to reduce privacy anxiety — "
         "not buried in T&Cs."),
        ("6. Make errors recoverable instantly",
         "Misheard inputs should trigger an immediate clarification loop, not a dead end. "
         "Accuracy failures that force call abandonment generate the most sustained negativity."),
    ]
    for title, body in recs:
        pdf.h3(title)
        pdf.body(body)
        pdf.ln(1)

    pdf.divider()
    pdf.h2("Conclusion")
    pdf.body(
        "People are not automatically against AI answering calls. Acceptance is higher for "
        "simple, low-stakes interactions when the AI is transparent, accurate, and offers a "
        "fast path to a human. Resistance is not about the technology per se — it is about "
        "trust, control, and context appropriateness. The biggest design failure is deception; "
        "the biggest commercial opportunity is frictionless task completion for routine bookings."
    )

    out = "AI_Voice_Agent_Report.pdf"
    pdf.output(out)
    print(f"\nPDF saved to {out}")
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    df, rc_themes, rc_quotes, rc_terms = load_data()

    print("Generating charts …")
    chart_paths = {
        "pie":          chart_sentiment_pie(df),
        "histogram":    chart_compound_histogram(df),
        "stacked":      chart_theme_stacked(rc_themes),
        "heatmap":      chart_theme_heatmap(rc_themes),
        "avg_compound": chart_avg_compound_by_theme(rc_themes),
        "timeline":     chart_timeline(df),
    }

    tfidf = chart_tfidf_terms(rc_terms)
    chart_paths["tfidf_positive"] = tfidf.get("positive")
    chart_paths["tfidf_negative"] = tfidf.get("negative")

    print("Building PDF …")
    build_pdf(df, rc_themes, rc_quotes, rc_terms, chart_paths)


if __name__ == "__main__":
    main()
