"""Central settings for the NLP Risk Engine. Every other file imports from here."""
import os
from pathlib import Path

try:                                    # optional: read secrets from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
RAW = DATA / "raw"                      # downloaded Kaggle / GDELT files
PROCESSED = DATA / "processed"          # engine output lives here

SIGNALS_PARQUET = PROCESSED / "signals.parquet"   # fast to load (dashboard uses this)
SIGNALS_CSV = PROCESSED / "signals.csv"           # human-readable copy for judges
SENTIMENT_CACHE = PROCESSED / "sentiment_cache.csv"   # scores already computed, keyed by signal id
ZEROSHOT_CACHE = PROCESSED / "zeroshot_cache.csv"     # event labels from the zero-shot model
ENGINE_META = PROCESSED / "engine_meta.json"          # what the last engine run produced
PRICES_CSV = PROCESSED / "prices.csv"
PORTFOLIO_CSV = PROCESSED / "portfolio.csv"

def ensure_dirs():
    for p in (RAW, PROCESSED):
        p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- run mode
# "live"   = fetch fresh data from the internet now (and cache it to data/raw)
# "replay" = play back the recent window recorded by earlier live runs
MODE = os.getenv("RISK_MODE", "replay")
REPLAY_DAYS = 3                         # Replay plays back the last 3 days of recorded signals
REPLAY_SPEEDS = [1, 2, 5, 10]           # speed control in the dashboard header

# ---------------------------------------------------------------- secrets
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Your Full Name your_id@college.ac.in")

# ---------------------------------------------------------------- server (src/api.py)
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
WEB_DIR = ROOT / "web"                  # the dashboard files go here; the server shows them at http://localhost:8000
AUTO_REFRESH = os.getenv("AUTO_REFRESH", "1") == "1"      # set AUTO_REFRESH=0 for a Replay-only demo
REFRESH_MINUTES = int(os.getenv("REFRESH_MINUTES", "30")) # each refresh makes ~4 NewsAPI requests; the free plan allows ~100 a day

# ---------------------------------------------------------------- data sources
# role:   "given" = listed in the problem statement, "extra" = our addition
# origin: "live"    = fetched from the internet, cached, and used by Live AND Replay
#         "dataset" = offline Kaggle file, used only to validate the sentiment model
# weight: how much we trust the source when scoring impact (0-1)
SOURCES = {
    "gdelt":       dict(role="given", kind="news",   weight=0.85, origin="live"),
    "newsapi":     dict(role="given", kind="news",   weight=0.90, origin="live"),
    "tweets":      dict(role="given", kind="social", weight=0.60, origin="dataset"),
    "kaggle_news": dict(role="given", kind="news",   weight=0.85, origin="dataset"),
    "sec_8k":      dict(role="extra", kind="filing", weight=1.00, origin="live"),
    "rss":         dict(role="extra", kind="news",   weight=0.85, origin="live"),
}
LIVE_SOURCES = [k for k, v in SOURCES.items() if v["origin"] == "live"]

# GDELT DOC API topic searches (market-wide news, one request every ~5 seconds)
GDELT_QUERIES = {
    "geopolitical": "(sanctions OR war OR missile OR tariffs OR ceasefire)",
    "macro":        '(inflation OR recession OR "interest rates" OR "central bank" OR GDP)',
    "credit":       '(default OR bankruptcy OR downgrade OR "debt crisis")',
    "mna":          "(merger OR acquisition OR takeover)",
    "regulatory":   '(antitrust OR lawsuit OR "SEC investigation")',
}

RSS_FEEDS = {                           # verify each URL opens before the demo
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol":   "https://www.moneycontrol.com/rss/business.xml",
}
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# ---------------------------------------------------------------- event taxonomy
# 5 from the problem statement + 2 extras. Value = base severity (1-10).
EVENTS = {
    "Geopolitical":       7,
    "Macroeconomic":      6,
    "Credit Event":       9,
    "Merger/Acquisition": 6,
    "Product Launch":     4,
    "Earnings":           5,   # extra
    "Regulatory/Legal":   6,   # extra
}
OTHER_EVENT = "Other"                   # fallback label, not a real category
EXTRA_EVENTS = ("Earnings", "Regulatory/Legal")

# SEC 8-K item number -> event (exact, so it beats keyword guessing)
SEC_ITEM_TO_EVENT = {
    "1.01": "Merger/Acquisition", "1.03": "Credit Event", "2.04": "Credit Event",
    "2.06": "Credit Event", "2.02": "Earnings", "3.01": "Regulatory/Legal",
}

# ---------------------------------------------------------------- scoring rules
SENT_POS = 0.15                         # score >= +0.15 -> Positive
SENT_NEG = -0.15                        # score <= -0.15 -> Negative, else Neutral
IMPACT_MIN, IMPACT_MAX = 1.0, 10.0
STRESS_TRIGGER_IMPACT = 7.0             # Module B fires a stress test at or above this
RISK_HALF_LIFE_H = 24                   # older signals count half as much every 24h
DEFAULT_WINDOW_H = 6                    # default window for the cards, donut and trend chart
HIGH_RISK_IMPACT = 9.0                  # banner shows HIGH RISK at this impact (ELEVATED at STRESS_TRIGGER_IMPACT)
RISK_K = 0.8                            # how fast the 0-1 asset risk score saturates
SPILLOVER_WEIGHT = 0.5                  # sector-tagged market news counts half for assets in that sector
FEED_SIZE = 50                          # rows in the left signal feed

# near-duplicate stories (same company/event, similar words, close in time) are merged into one
DEDUP_SIM = 0.60                        # share of words two headlines must have in common
DEDUP_WINDOW_H = 6

# event classification (rules first, zero-shot model only for unclear texts)
ZEROSHOT_MODEL = "typeform/distilbert-base-uncased-mnli"   # small, CPU-friendly; swap for
                                                           # "facebook/bart-large-mnli" if you want accuracy over speed
ZS_MIN_SCORE = 0.45                     # below this the zero-shot answer is treated as "Other"

# impact is scored relative to recent news (see events.py)
IMPACT_WINDOW_H = 72                    # compare each signal with the last 72 hours...
IMPACT_MIN_N = 30                       # ...or the last 30 signals if that is more
IMPACT_TOP_SHARE = 0.10                 # about the top 10% of signals reach STRESS_TRIGGER_IMPACT

# sentiment model settings (tuned for a CPU-only laptop)
FINBERT_MODEL = "ProsusAI/finbert"
SENT_BATCH = 16                         # texts scored at once
SENT_MAX_LEN = 96                       # max tokens per text (headlines are short)
SENT_MAX_CHARS = 600                    # cut very long texts before tokenizing

def label_sentiment(score: float) -> str:
    if score >= SENT_POS:
        return "Positive"
    if score <= SENT_NEG:
        return "Negative"
    return "Neutral"

# ---------------------------------------------------------------- company matching (hybrid)
MARKET_ENTITY = "MARKET"                # text not about one company
SPACY_MODEL = "en_core_web_sm"
NER_LABELS = ("ORG",)                   # spaCy labels treated as company names (used only to confirm/spot watchlist companies)

SECTORS = ["Technology", "Financials", "Consumer", "Energy", "Healthcare", "Others"]

# 15 S&P 100 stocks. yahoo = ticker used by yfinance. aliases = words that identify it.
WATCHLIST = {
    "AAPL":  dict(yahoo="AAPL",  sector="Technology", aliases=["apple", "iphone"]),
    "MSFT":  dict(yahoo="MSFT",  sector="Technology", aliases=["microsoft", "azure"]),
    "NVDA":  dict(yahoo="NVDA",  sector="Technology", aliases=["nvidia"]),
    "META":  dict(yahoo="META",  sector="Technology", aliases=["meta platforms", "facebook", "instagram"]),
    "AMZN":  dict(yahoo="AMZN",  sector="Consumer",   aliases=["amazon", "aws"]),
    "TSLA":  dict(yahoo="TSLA",  sector="Consumer",   aliases=["tesla"]),
    "WMT":   dict(yahoo="WMT",   sector="Consumer",   aliases=["walmart"]),
    "KO":    dict(yahoo="KO",    sector="Consumer",   aliases=["coca-cola", "coca cola"]),
    "JPM":   dict(yahoo="JPM",   sector="Financials", aliases=["jpmorgan", "jp morgan"]),
    "BAC":   dict(yahoo="BAC",   sector="Financials", aliases=["bank of america"]),
    "GS":    dict(yahoo="GS",    sector="Financials", aliases=["goldman sachs"]),
    "BRK.B": dict(yahoo="BRK-B", sector="Financials", aliases=["berkshire hathaway", "berkshire"]),
    "XOM":   dict(yahoo="XOM",   sector="Energy",     aliases=["exxon", "exxonmobil"]),
    "JNJ":   dict(yahoo="JNJ",   sector="Healthcare", aliases=["johnson & johnson"]),
    "PFE":   dict(yahoo="PFE",   sector="Healthcare", aliases=["pfizer"]),
}
ALIAS_TO_TICKER = {a: t for t, c in WATCHLIST.items() for a in c["aliases"]}
TICKER_TO_SECTOR = {t: c["sector"] for t, c in WATCHLIST.items()}

# ---------------------------------------------------------------- the signal contract
# Every row the engine writes has exactly these columns. Modules A, B and the
# dashboard only ever read these, so the engine stays swappable.
SIGNAL_COLUMNS = [
    "signal_id",    # unique id (hash of source + text)
    "ts",           # UTC timestamp of the original item
    "source",       # key of SOURCES
    "entity",       # watchlist ticker, or MARKET for everything else
    "entity_type",  # "watchlist" | "market"
    "sector",       # one of SECTORS
    "event",        # one of EVENTS or Other
    "sentiment",    # -1.0 .. +1.0
    "sentiment_label",  # Positive / Neutral / Negative
    "impact",       # 1 .. 10
    "text",         # headline / snippet
]

# ---------------------------------------------------------------- rebalancer (Module A)
REBAL_FREQ_H = 1                        # rebalance every hour
REBAL_TILT = 1.0                        # weight = start x (1 + TILT x score). Keep at 1.0 or below so weights never go negative
REBAL_HALF_LIFE_H = 12                  # a headline counts half as much after 12 hours
REBAL_SHRINK = 0.5                      # pulls a stock's score towards 0 when it has little news
REBAL_MIN_W, REBAL_MAX_W = 0.02, 0.12   # every stock stays between 2% and 12%
REBAL_TURNOVER_CAP = 0.10               # at most 10% of the portfolio is traded per rebalance

# ---------------------------------------------------------------- market prices (Module A)
PRICE_INTERVAL = "1d"                   # daily closes; "1h" gives hourly bars (Yahoo keeps about 2 years of them)
PRICE_PERIOD = "6mo"                    # how much history to download
CLOSE_HOUR_UTC = 21                     # a daily close counts as known from 21:00 UTC (after the US close)

# ---------------------------------------------------------------- Module B portfolio
PORTFOLIO_SEED = 7
PORTFOLIO_SIZE = 100
PORTFOLIO_TOTAL_USD = 1_500_000_000       # about US$1.5 billion in total