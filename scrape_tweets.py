"""
Scrape UK AI voice agent tweets via twitterapi.io advanced_search.

Mirrors x-marketing/src/search.ts exactly:
  - GET https://api.twitterapi.io/twitter/tweet/advanced_search
  - Header: X-API-Key (env var: API_KEY)
  - Params: query, queryType, cursor
  - Operators: -filter:replies -filter:retweets appended to every query
  - Paginates via has_next_page / next_cursor
  - No proxy needed for search (only used for posting)

Field names are exact camelCase from the API (see src/response.json):
  id, text, createdAt, lang, likeCount, retweetCount, replyCount,
  quoteCount, viewCount, bookmarkCount, isReply
  author.userName, author.location, author.isBlueVerified, author.followers

Usage:
    python scrape_tweets.py [--max-per-query 100] [--out tweets_raw.json]
"""

import os
import json
import time
import argparse
from pathlib import Path

import csv

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
SEARCH_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
OPERATORS = "-filter:replies -filter:retweets"

QUERIES = [
    '"AI customer service" (UK OR Britain OR London)',
    '"AI voice" "customer service" (UK OR Britain OR London)',
    '"AI receptionist" (UK OR Britain OR London)',
    '"automated phone system" (UK OR Britain OR London)',
    '"robot voice" "customer service"',
    '"can\'t speak to a human"',
    '"talking to a bot" (UK OR Britain OR London)',
    '"AI answered the phone"',
    '"human sounding AI"',
    '"Google Duplex"',
    '"voice bot" (UK OR Britain OR London)',
    '"phone bot"',
    '"virtual receptionist" (UK OR Britain OR London)',
    '"AI phone call"',
    '"automated booking" (UK OR Britain OR London)',
    '"GP appointment" (bot OR AI)',
    'NHS (AI phone OR AI voice OR voice bot)',
    'taxi (AI phone OR voice bot) (UK OR London)',
    '"restaurant booking" (AI OR bot)',
]


def fetch_tweets(query: str, max_tweets: int, query_type: str = "Latest") -> list[dict]:
    """Paginate twitterapi.io exactly as search.ts does."""
    full_query = f"{query} {OPERATORS}".strip()
    collected: list[dict] = []
    cursor = ""

    while len(collected) < max_tweets:
        params = {"query": full_query, "queryType": query_type, "cursor": cursor}
        try:
            resp = requests.get(
                SEARCH_URL,
                headers={"X-API-Key": API_KEY},
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [warn] request error: {e}")
            break

        if data.get("status") != "success" and "tweets" not in data:
            print(f"  [warn] unexpected response: {data.get('message', data)}")
            break

        for t in data.get("tweets", []):
            author = t.get("author") or {}
            collected.append({
                "tweet_id":       t["id"],
                "text":           t.get("text", ""),
                "created_at":     t.get("createdAt", ""),
                "lang":           t.get("lang", ""),
                "like_count":     t.get("likeCount", 0),
                "retweet_count":  t.get("retweetCount", 0),
                "reply_count":    t.get("replyCount", 0),
                "quote_count":    t.get("quoteCount", 0),
                "view_count":     t.get("viewCount", 0),
                "bookmark_count": t.get("bookmarkCount", 0),
                "is_reply":       t.get("isReply", False),
                "username":       author.get("userName", ""),
                "user_location":  author.get("location", ""),
                "blue_verified":  author.get("isBlueVerified", False),
                "followers":      author.get("followers", 0),
                "query_used":     query,
            })

        if not data.get("has_next_page"):
            break
        cursor = data.get("next_cursor", "")
        time.sleep(1.0)

    return collected[:max_tweets]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-query", type=int, default=100)
    parser.add_argument("--query-type", default="Latest", choices=["Latest", "Top"])
    parser.add_argument("--out", default="tweets_raw.json")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("ERROR: API_KEY not set in .env")

    # Mirror x-marketing: dump raw JSON to logs/
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    all_tweets: list[dict] = []
    seen_ids: set[str] = set()

    for i, query in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {query!r}")
        results = fetch_tweets(query, max_tweets=args.max_per_query, query_type=args.query_type)

        # dump raw per query (mirrors pipeline.ts step 2)
        safe_name = "".join(c if c.isalnum() else "_" for c in query)[:60]
        raw_path = log_dir / f"raw_{safe_name}.json"
        raw_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        new = [t for t in results if t["tweet_id"] not in seen_ids]
        seen_ids.update(t["tweet_id"] for t in new)
        all_tweets.extend(new)
        print(f"  => fetched {len(results)}, {len(new)} new (total unique: {len(all_tweets)})")
        time.sleep(2.0)

    # Save raw JSON
    out_path = Path(args.out)
    out_path.write_text(json.dumps(all_tweets, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(all_tweets)} unique tweets to {args.out}")

    # Save a standalone CSV of raw tweets (no NLP columns)
    tweets_csv = Path("tweets.csv")
    if all_tweets:
        with open(tweets_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_tweets[0].keys())
            writer.writeheader()
            writer.writerows(all_tweets)
    print(f"Saved raw tweet CSV to {tweets_csv}")


if __name__ == "__main__":
    main()
