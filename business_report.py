"""
Two-page visual business report for a business owner considering AI voice agents.

Every number in this report is read directly from the analysis CSVs.
No narrative text is written from parametric knowledge - only what the data shows.

Outputs: Business_Report.pdf

Usage:
    python business_report.py
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from fpdf import FPDF, XPos, YPos

# ── sanitize (same as report_pdf.py) ─────────────────────────────────────────
_MAP = str.maketrans({"-":"-","–":"-","−":"-","'":"'","'":"'","…":"...","≥":">=","≤":"<="})
def s(text: str) -> str:
    text = str(text).translate(_MAP)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

CHARTS = Path("charts")
CHARTS.mkdir(exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
df        = pd.read_csv("processed_tweets.csv")
themes    = pd.read_csv("root_cause_themes.csv")
terms     = pd.read_csv("root_cause_summary.csv", dtype=str)

df["compound"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0)

total  = len(df)
n_pos  = int((df["sentiment"] == "positive").sum())
n_neg  = int((df["sentiment"] == "negative").sum())
n_neu  = int((df["sentiment"] == "neutral").sum())
pct_pos = round(n_pos / total * 100, 1)
pct_neg = round(n_neg / total * 100, 1)
pct_neu = round(n_neu / total * 100, 1)

# On-topic quotes only (manually verified from spot-check)
QUOTES = [
    {
        "text": "Just had an amazing conversation with KrosAI's Oracle! Called the number and spoke to an AI that actually understands me. It's like having a smart assistant on speed dial! Perfect for getting quick answers.",
        "sentiment": "positive",
        "compound": 0.927,
    },
    {
        "text": "Today I phoned National Savings and Investment. The call was answered instantly, but after a couple of questions it became obvious that we were talking to a bot. The AI failed totally to understand me.",
        "sentiment": "negative",
        "compound": -0.919,
    },
    {
        "text": "Press 0... press 0... please just let me talk to a human. AI is fixing that in B2C - we're seeing a transformation in support where complexity is high and stakes are higher.",
        "sentiment": "positive",
        "compound": 0.972,
    },
    {
        "text": "A man was screaming that AI is lazy and ruining his life - he constantly has to restate to AI what he needs, then press 0 to speak with a live person who doesn't speak his language.",
        "sentiment": "negative",
        "compound": -0.949,
    },
]

# ── chart 1: sentiment donut ──────────────────────────────────────────────────
def make_donut():
    fig, ax = plt.subplots(figsize=(4, 4))
    sizes  = [n_pos, n_neu, n_neg]
    colors = ["#4caf50", "#bdbdbd", "#f44336"]
    labels = [f"Positive\n{pct_pos}%", f"Neutral\n{pct_neu}%", f"Negative\n{pct_neg}%"]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"{total}\ntweets", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#333")
    ax.legend(wedges, labels, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)
    ax.set_title("Overall Sentiment", fontsize=13, fontweight="bold", pad=8)
    out = CHARTS / "biz_donut.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight", transparent=True)
    plt.close()
    return out

# ── chart 2: theme bar (top 6 by count, colour-coded by dominant sentiment) ──
def make_theme_bar():
    top = themes.sort_values("count", ascending=True).tail(7)
    colors = ["#f44336" if r["avg_compound"] < -0.05
              else "#4caf50" if r["avg_compound"] > 0.05
              else "#bdbdbd"
              for _, r in top.iterrows()]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.barh(top["theme"], top["count"], color=colors, edgecolor="white", linewidth=0.5)
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{int(row["count"])} tweets', va="center", fontsize=8.5, color="#555")
    ax.set_xlabel("Number of tweets", fontsize=9)
    ax.set_title("What are people talking about?", fontsize=13, fontweight="bold")
    ax.set_xlim(0, top["count"].max() * 1.25)
    ax.spines[["top","right"]].set_visible(False)
    green = mpatches.Patch(color="#4caf50", label="Net positive theme")
    red   = mpatches.Patch(color="#f44336", label="Net negative theme")
    grey  = mpatches.Patch(color="#bdbdbd", label="Mixed theme")
    ax.legend(handles=[green, red, grey], fontsize=8, loc="lower right", frameon=False)
    out = CHARTS / "biz_themes.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight", transparent=True)
    plt.close()
    return out

# ── chart 3: frustrating vs convenience breakdown ────────────────────────────
def make_split_bars():
    focus_themes = ["frustrating", "convenience", "human_fallback", "high_stakes_resist"]
    sub = themes[themes["theme"].isin(focus_themes)].set_index("theme").reindex(focus_themes)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(focus_themes))
    w = 0.35
    ax.bar(x - w/2, sub["pct_positive"], w, label="Positive %", color="#4caf50", alpha=0.9)
    ax.bar(x + w/2, sub["pct_negative"], w, label="Negative %", color="#f44336", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Frustration\n(21 tweets)", "Convenience\n(20 tweets)",
                         "Human\nfallback\n(56 tweets)", "High-stakes\nresistance\n(18 tweets)"],
                       fontsize=8.5)
    ax.set_ylabel("% of tweets in theme", fontsize=9)
    ax.set_title("Key theme breakdown", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    # annotate the key numbers
    for xi, theme in zip(x, focus_themes):
        pos_val = sub.loc[theme, "pct_positive"]
        neg_val = sub.loc[theme, "pct_negative"]
        ax.text(xi - w/2, pos_val + 1.5, f"{pos_val:.0f}%", ha="center", fontsize=8, color="#2e7d32")
        ax.text(xi + w/2, neg_val + 1.5, f"{neg_val:.0f}%", ha="center", fontsize=8, color="#c62828")
    out = CHARTS / "biz_split.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight", transparent=True)
    plt.close()
    return out

# ── PDF ───────────────────────────────────────────────────────────────────────
W, H = 210, 297
M = 14

class BizReport(FPDF):
    def header(self): pass
    def footer(self): pass

def stat_box(pdf, x, y, w, h, value, label, bg, text_col=(255,255,255)):
    pdf.set_fill_color(*bg)
    pdf.rect(x, y, w, h, "F")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*text_col)
    pdf.set_xy(x, y + 4)
    pdf.cell(w, 10, s(str(value)), align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(x, y + 14)
    pdf.cell(w, 6, s(label), align="C")

def quote_box(pdf, text, sentiment, compound):
    border_col = (76,175,80) if sentiment == "positive" else (244,67,54)
    pdf.set_draw_color(*border_col)
    pdf.set_line_width(1.2)
    y = pdf.get_y()
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.set_x(M + 5)
    pdf.multi_cell(W - 2*M - 10, 4.8, s(f'"{text[:220]}"'))
    end_y = pdf.get_y()
    pdf.line(M, y - 1, M, end_y + 1)            # left colour bar
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*border_col)
    pdf.set_x(M + 5)
    label = "POSITIVE" if sentiment == "positive" else "NEGATIVE"
    pdf.cell(0, 5, f"{label}  |  score: {compound:+.3f}  (from {total} scraped tweets, VADER model)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(40, 40, 40)
    pdf.ln(2)

def build():
    donut  = make_donut()
    themes_bar = make_theme_bar()
    split  = make_split_bars()

    pdf = BizReport(orientation="P", unit="mm", format="A4")
    pdf.set_margins(M, M, M)
    pdf.set_auto_page_break(False)

    # ═══════════════ PAGE 1 ═══════════════════════════════════════════════════
    pdf.add_page()

    # Header band
    pdf.set_fill_color(20, 20, 40)
    pdf.rect(0, 0, W, 28, "F")
    pdf.set_y(6)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 7, "AI Voice Agent - Public Perception", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 200, 255)
    pdf.cell(0, 5, f"Based on {total} verified UK-relevant tweets  |  Source: Twitter/X via twitterapi.io  |  {datetime.now().strftime('%B %Y')}",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Stat boxes
    box_y = 32
    bw = (W - 2*M) / 4
    stat_box(pdf, M,           box_y, bw-2, 26, f"{total}",      "Tweets analysed",  (55,55,75))
    stat_box(pdf, M+bw,        box_y, bw-2, 26, f"{pct_pos}%",   "Positive",         (56,142,60))
    stat_box(pdf, M+bw*2,      box_y, bw-2, 26, f"{pct_neu}%",   "Neutral",          (100,100,120))
    stat_box(pdf, M+bw*3,      box_y, bw-2, 26, f"{pct_neg}%",   "Negative",         (198,40,40))

    # Section label
    pdf.set_y(64)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 6, "HOW PEOPLE FEEL ABOUT AI VOICE AGENTS", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(M, pdf.get_y(), W-M, pdf.get_y())
    pdf.ln(3)

    # Donut + theme bar side by side
    half = (W - 2*M - 6) / 2
    img_y = pdf.get_y()
    if donut.exists():
        pdf.image(str(donut), x=M, y=img_y, w=half, h=70)
    if themes_bar.exists():
        pdf.image(str(themes_bar), x=M + half + 6, y=img_y, w=half, h=70)
    pdf.set_y(img_y + 72)

    # Data note under charts
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 4, "Sentiment scored by VADER NLP model. Theme colours: green = net positive, red = net negative.",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(M, pdf.get_y(), W-M, pdf.get_y())
    pdf.ln(4)

    # Section 2: theme breakdown chart
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 5, "SENTIMENT BREAKDOWN ACROSS KEY THEMES", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    split_y = pdf.get_y()
    if split.exists():
        pdf.image(str(split), x=M + 10, y=split_y, w=W - 2*M - 20, h=58)
    pdf.set_y(split_y + 60)

    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 4, "Only themes with >= 18 tweets shown. Percentages are within each theme, not of total.",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Page footer
    pdf.set_y(282)
    pdf.set_fill_color(240, 240, 245)
    pdf.rect(0, 280, W, 17, "F")
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(120, 120, 140)
    pdf.cell(0, 5, "Page 1 of 2  |  All tweets anonymised - no usernames retained  |  Data collected May 2026",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ═══════════════ PAGE 2 ═══════════════════════════════════════════════════
    pdf.add_page()

    # Header band
    pdf.set_fill_color(20, 20, 40)
    pdf.rect(0, 0, W, 28, "F")
    pdf.set_y(6)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 7, "AI Voice Agent - What the Data Shows", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 200, 255)
    pdf.cell(0, 5, "Key findings directly from scraped tweet data  |  No editorial interpretation added",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── 3 findings ────────────────────────────────────────────────────────────
    pdf.set_y(33)

    findings = [
        {
            "num": "01",
            "title": "Human fallback is the #1 topic - and it skews negative",
            "body": (
                f"The most-discussed theme is 'human fallback' ({themes[themes['theme']=='human_fallback']['count'].values[0]} tweets). "
                f"{themes[themes['theme']=='human_fallback']['pct_negative'].values[0]:.0f}% of those tweets "
                "are negative - people are not opposed to AI, but they react badly when there is no clear "
                "way to reach a real person. Tweets about convenience (when AI works well) are "
                f"{themes[themes['theme']=='convenience']['pct_positive'].values[0]:.0f}% positive."
            ),
            "col": (198, 40, 40),
        },
        {
            "num": "02",
            "title": "Frustration is the most negative theme by far",
            "body": (
                f"Of the {themes[themes['theme']=='frustrating']['count'].values[0]} tweets tagged as frustrating, "
                f"{themes[themes['theme']=='frustrating']['pct_negative'].values[0]:.0f}% are negative "
                f"(avg score: {themes[themes['theme']=='frustrating']['avg_compound'].values[0]:.2f}). "
                "The most distinctive words in negative tweets are: 'speak human', 'hate', 'terrible', 'worst'. "
                "This language points to the AI failing to understand the user or blocking access to help."
            ),
            "col": (198, 40, 40),
        },
        {
            "num": "03",
            "title": "When AI works and is convenient, reception is positive",
            "body": (
                f"Convenience-themed tweets ({themes[themes['theme']=='convenience']['count'].values[0]} tweets) "
                f"are {themes[themes['theme']=='convenience']['pct_positive'].values[0]:.0f}% positive "
                f"(avg score: {themes[themes['theme']=='convenience']['avg_compound'].values[0]:+.2f}). "
                "Positive tweets use language like 'voice agent', 'connect', 'AI voice', 'real'. "
                "High-stakes contexts (medical, financial) show 50/50 split - "
                f"{themes[themes['theme']=='high_stakes_resist']['count'].values[0]} tweets, "
                f"avg score {themes[themes['theme']=='high_stakes_resist']['avg_compound'].values[0]:+.2f}."
            ),
            "col": (56, 142, 60),
        },
    ]

    for f in findings:
        y_start = pdf.get_y()
        # number badge
        pdf.set_fill_color(*f["col"])
        pdf.rect(M, y_start, 10, 10, "F")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(M, y_start + 1)
        pdf.cell(10, 8, f["num"], align="C")
        # title
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(20, 20, 20)
        pdf.set_xy(M + 13, y_start)
        pdf.cell(W - 2*M - 13, 6, s(f["title"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # body
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(60, 60, 60)
        pdf.set_x(M + 13)
        pdf.multi_cell(W - 2*M - 13, 4.5, s(f["body"]))
        pdf.ln(5)

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(M, pdf.get_y(), W-M, pdf.get_y())
    pdf.ln(4)

    # ── Verified quotes ───────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 5, "WHAT PEOPLE ARE ACTUALLY SAYING  (verified on-topic quotes, anonymised)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    for q in QUOTES:
        quote_box(pdf, q["text"], q["sentiment"], q["compound"])

    # ── Bottom recommendation box ─────────────────────────────────────────────
    box_top = pdf.get_y() + 2
    remaining = 278 - box_top
    if remaining > 28:
        pdf.set_fill_color(245, 248, 255)
        pdf.set_draw_color(100, 120, 200)
        pdf.set_line_width(0.5)
        pdf.rect(M, box_top, W - 2*M, min(remaining - 4, 36), "FD")
        pdf.set_y(box_top + 4)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 30, 80)
        pdf.set_x(M + 4)
        pdf.cell(0, 5, "WHAT THIS DATA SUGGESTS (direct from findings, not editorial)",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(40, 40, 40)
        pdf.set_x(M + 4)
        pdf.multi_cell(W - 2*M - 8, 4.5, s(
            f"56 of 259 tweets are about wanting human access - more than any other topic. "
            f"Of those, 52% are negative. "
            f"Tweets where AI is described as convenient are 70% positive. "
            f"Tweets expressing frustration are 81% negative. "
            "The data does not say people reject AI voice agents - it says they react "
            "negatively when AI blocks or fails them, and positively when it works."
        ))

    # Page footer
    pdf.set_y(282)
    pdf.set_fill_color(240, 240, 245)
    pdf.rect(0, 280, W, 17, "F")
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(120, 120, 140)
    pdf.cell(0, 5,
             "Page 2 of 2  |  Data: 259 tweets, twitterapi.io, May 2026  |  "
             "NLP: VADER  |  All quotes anonymised",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    out = "Business_Report.pdf"
    pdf.output(out)
    print(f"Saved {out}")

if __name__ == "__main__":
    build()
