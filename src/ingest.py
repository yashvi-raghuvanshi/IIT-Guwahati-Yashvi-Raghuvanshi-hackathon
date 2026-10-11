"""Ingestion: pull text from every source, cache it, return ONE uniform table.

Live mode   -> fetch from the internet, save to data/raw/<source>.csv, return fresh rows.
Replay mode -> read those saved files (the recent window recorded by live runs).
Kaggle files are offline datasets used only to validate the sentiment model.
"""
import hashlib
import re
import time
from calendar import timegm
from datetime import datetime, timezone

import feedparser
import pandas as pd
import requests

from src import config as C

RECORD_COLS = ["id", "ts", "source", "text", "ticker_hint", "sec_items"]

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (risk-engine hackathon prototype)"


# ------------------------------------------------------------------ helpers
def _clean(html):
    """Strip HTML tags and extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(html or ""))).strip()


def _rec(source, text, ts, ticker=None, items=None):
    text = _clean(text)
    uid = hashlib.md5(f"{source}|{text}".encode()).hexdigest()
    return dict(id=uid, ts=ts, source=source, text=text, ticker_hint=ticker, sec_items=items)


def _feed_ts(entry):
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime.fromtimestamp(timegm(t), tz=timezone.utc) if t else datetime.now(timezone.utc)


def _get(url, retries=2, **kw):
    """requests.get with a timeout and a couple of retries. After a 429 ("too many requests") it waits much longer."""
    for i in range(retries + 1):
        try:
            r = _session.get(url, timeout=20, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as ex:
            if i == retries:
                raise
            wait = 1.5 * (i + 1)
            resp = getattr(ex, "response", None)
            if resp is not None and resp.status_code == 429:
                try:
                    wait = float(resp.headers.get("Retry-After") or 0) or 8.0 * (i + 1)
                except ValueError:
                    wait = 8.0 * (i + 1)
            time.sleep(wait)


def _to_df(rows):
    df = pd.DataFrame(rows, columns=RECORD_COLS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates("id")


# ------------------------------------------------------------------ live sources
def fetch_gdelt(timespan="1d", per_query=50):
    """GDELT DOC API: global headlines for our topic searches (no key needed)."""
    rows = []
    for name, q in C.GDELT_QUERIES.items():
        try:
            r = _get("https://api.gdeltproject.org/api/v2/doc/doc", params=dict(
                query=f"{q} sourcelang:english", mode="artlist", format="json",
                maxrecords=per_query, sort="datedesc", timespan=timespan))
            try:
                articles = r.json().get("articles", [])
            except ValueError:                  # GDELT sometimes answers with plain text
                articles = []
            for a in articles:
                ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                rows.append(_rec("gdelt", a["title"], ts))
        except Exception as ex:                 # one failed search must not throw away the others
            print(f"[warn] gdelt search '{name}' failed ({type(ex).__name__}); skipping it")
        time.sleep(5.2)                         # GDELT allows ~1 request per 5 seconds
    return rows


def fetch_newsapi(batch=5):
    """NewsAPI: company headlines in batches (saves the free daily request limit)."""
    if not C.NEWSAPI_KEY:
        print("[info] NEWSAPI_KEY not set, skipping newsapi")
        return []
    hdr = {"X-Api-Key": C.NEWSAPI_KEY}
    tickers, rows = list(C.WATCHLIST), []
    for i in range(0, len(tickers), batch):
        names = [C.WATCHLIST[t]["aliases"][0] for t in tickers[i:i + batch]]
        q = " OR ".join(f'"{n}"' for n in names)
        j = _get("https://newsapi.org/v2/everything", headers=hdr,
                 params=dict(q=q, language="en", sortBy="publishedAt", pageSize=50)).json()
        for a in j.get("articles", []):
            rows.append(_rec("newsapi", f"{a['title']}. {a.get('description') or ''}",
                             pd.to_datetime(a["publishedAt"], utc=True)))
    j = _get("https://newsapi.org/v2/top-headlines", headers=hdr,
             params=dict(category="business", country="us", pageSize=50)).json()
    for a in j.get("articles", []):
        rows.append(_rec("newsapi", f"{a['title']}. {a.get('description') or ''}",
                         pd.to_datetime(a["publishedAt"], utc=True)))
    return rows


def fetch_rss(per_company=15, per_feed=40):
    """Extra source: Google News (per company) + Economic Times + Moneycontrol."""
    rows = []
    for tk, cfg in C.WATCHLIST.items():
        url = C.GOOGLE_NEWS_RSS.format(query=requests.utils.quote(f"{cfg['aliases'][0]} stock"))
        for e in feedparser.parse(url).entries[:per_company]:
            rows.append(_rec("rss", e.get("title", ""), _feed_ts(e), ticker=tk))
    for url in C.RSS_FEEDS.values():
        for e in feedparser.parse(url).entries[:per_feed]:
            rows.append(_rec("rss", f"{e.get('title', '')}. {e.get('summary', '')}", _feed_ts(e)))
    return rows


def fetch_edgar_8k(per_company=8):
    """Extra source: SEC EDGAR 8-K filings. The item numbers tell us the event type."""
    rows = []
    for tk, cfg in C.WATCHLIST.items():
        url = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
               f"&CIK={cfg['yahoo']}&type=8-K&owner=include&count={per_company}&output=atom")
        r = _get(url, headers={"User-Agent": C.SEC_USER_AGENT})
        for e in feedparser.parse(r.content).entries:
            text = _clean(e.get("summary", ""))
            items = ",".join(re.findall(r"Item (\d\.\d\d)", text))
            rows.append(_rec("sec_8k", f"{tk} 8-K filing. {text}", _feed_ts(e), ticker=tk, items=items))
        time.sleep(0.2)                         # SEC allows ~10 requests per second
    return rows


LIVE_FETCHERS = {"gdelt": fetch_gdelt, "newsapi": fetch_newsapi,
                 "sec_8k": fetch_edgar_8k, "rss": fetch_rss}


# ------------------------------------------------------------------ cache (data/raw/<source>.csv)
def _cache_path(source):
    return C.RAW / f"{source}.csv"


def _save_cache(source, df):
    C.ensure_dirs()
    p = _cache_path(source)
    if p.exists():
        df = pd.concat([_load_cache(source), df])
    df.drop_duplicates("id").to_csv(p, index=False)


def _load_cache(source):
    p = _cache_path(source)
    if not p.exists():
        return _to_df([])
    df = pd.read_csv(p, dtype={"id": str, "ticker_hint": "object", "sec_items": "object"})  # ids stay text
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# ------------------------------------------------------------------ public entry points
def fetch_live():
    """Fetch every live source, cache it, return the fresh rows.
    If one source fails, fall back to its cache so the demo never breaks."""
    frames = []
    for source, fn in LIVE_FETCHERS.items():
        try:
            df = _to_df(fn())
            if df.empty:
                raise RuntimeError("no rows returned")
            _save_cache(source, df)
        except Exception as ex:
            print(f"[warn] {source}: live fetch failed ({ex}); using cache")
            df = _load_cache(source)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates("id")


def load_replay(days=None):
    """Read everything recorded by earlier live runs (optionally only the last N days)."""
    df = pd.concat([_load_cache(s) for s in C.LIVE_SOURCES], ignore_index=True).drop_duplicates("id")
    if days and not df.empty:
        df = df[df.ts >= df.ts.max() - pd.Timedelta(days=days)]
    return df.sort_values("ts").reset_index(drop=True)


def fetch_all(mode=None):
    return fetch_live() if (mode or C.MODE) == "live" else load_replay()


# ------------------------------------------------------------------ Kaggle datasets (validation only)
def load_kaggle_tweets(path=None, n=2000):
    """'Stock Tweets for Sentiment Analysis and Prediction' (columns: Date, Tweet, Stock Name)."""
    path = path or C.RAW / "stock_tweets.csv"
    if not path.exists():
        return _to_df([])
    df = pd.read_csv(path).dropna(subset=["Tweet"]).tail(n)
    by_yahoo = {c["yahoo"]: t for t, c in C.WATCHLIST.items()}
    rows = [_rec("tweets", r["Tweet"], pd.to_datetime(r["Date"], utc=True),
                 ticker=by_yahoo.get(str(r.get("Stock Name", ""))))
            for _, r in df.iterrows()]
    return _to_df(rows)


def _read_csv_any_encoding(path, **kw):
    """UTF-8 first (also handles a BOM), then Latin-1, which is how the original Kaggle file is saved."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kw)


def load_kaggle_news(path=None):
    """'Sentiment Analysis for Financial News' -> DataFrame[text, label] with positive/neutral/negative."""
    path = path or C.RAW / "financial_news.csv"
    if not path.exists():
        return pd.DataFrame(columns=["text", "label"])
    df = _read_csv_any_encoding(path)
    low = {c.lower(): c for c in df.columns}
    if not ({"sentence", "text"} & set(low)):               # file has no header row
        df = _read_csv_any_encoding(path, header=None, names=["label", "text"])
        low = {"label": "label", "text": "text"}
    text_col = low.get("sentence") or low["text"]
    label_col = low.get("sentiment") or low["label"]
    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out["label"] = out["label"].str.lower().str.strip()
    return out.dropna()


if __name__ == "__main__":
    data = fetch_live()
    print(data.groupby("source").size().to_string())
    print("Time range:", data.ts.min(), "->", data.ts.max())