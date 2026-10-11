"""Sentiment: turn text into a score from -1.0 (negative) to +1.0 (positive).

Uses FinBERT (trained on financial text). If it can't load, falls back to VADER.
Built for a CPU-only laptop: short max length, batching, sorted by length,
and every score is cached by signal id so old items are never scored twice.
"""
import pandas as pd

from src import config as C


# ------------------------------------------------------------------ the two models
class _FinBert:
    name = "finbert"

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=C.FINBERT_MODEL,
                             top_k=None, device=-1)          # device=-1 means CPU

    def score(self, texts):
        out = self.pipe(texts, batch_size=C.SENT_BATCH, truncation=True, max_length=C.SENT_MAX_LEN)
        scores = []
        for result in out:
            p = {d["label"].lower(): d["score"] for d in result}
            scores.append(p.get("positive", 0.0) - p.get("negative", 0.0))
        return scores


class _Vader:
    name = "vader"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.sia = SentimentIntensityAnalyzer()

    def score(self, texts):
        return [self.sia.polarity_scores(t)["compound"] for t in texts]


def make_backend(name):
    return {"finbert": _FinBert, "vader": _Vader}[name]()


_backend = None


def get_backend():
    """Load FinBERT once; fall back to VADER if it fails (no internet, no torch...)."""
    global _backend
    if _backend is None:
        try:
            _backend = make_backend("finbert")
        except Exception as ex:
            print(f"[warn] FinBERT unavailable ({type(ex).__name__}); falling back to VADER")
            try:
                _backend = make_backend("vader")
            except Exception as ex2:
                raise RuntimeError("No sentiment model available. "
                                   "Run: pip install transformers torch vaderSentiment") from ex2
    return _backend


# ------------------------------------------------------------------ scoring
def score_texts(texts, backend=None, show_progress=True, chunk=64):
    """Score a list of texts. Returns scores in the SAME order as the input."""
    be = backend or get_backend()
    texts = [str(t)[:C.SENT_MAX_CHARS] for t in texts]
    scores = [0.0] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # similar lengths = less padding
    for start in range(0, len(order), chunk):
        idx = order[start:start + chunk]
        for i, v in zip(idx, be.score([texts[i] for i in idx])):
            scores[i] = round(float(v), 4)
        if show_progress:
            print(f"  scored {min(start + chunk, len(order))}/{len(order)} ({be.name})", end="\r")
    if show_progress and texts:
        print()
    return scores


# ------------------------------------------------------------------ cache
def _read_cache():
    if C.SENTIMENT_CACHE.exists():
        return pd.read_csv(C.SENTIMENT_CACHE, dtype={"id": str})   # ids must stay text
    return pd.DataFrame(columns=["id", "sentiment", "model"])


def score_df(df):
    """Score a table that has 'id' and 'text' columns. Only NEW items are scored.
    Returns a Series of sentiment values aligned with df."""
    be = get_backend()
    cache = _read_cache()
    done = cache[cache.model == be.name].drop_duplicates("id", keep="last").set_index("id").sentiment  # ignore other models
    todo = df[~df.id.isin(done.index)].drop_duplicates("id")
    if len(todo):
        new = pd.DataFrame({"id": todo.id.values,
                            "sentiment": score_texts(todo.text.tolist(), be),
                            "model": be.name})
        C.ensure_dirs()
        pd.concat([cache, new]).drop_duplicates("id", keep="last").to_csv(C.SENTIMENT_CACHE, index=False)
        done = pd.concat([done, new.set_index("id").sentiment])
    return df.id.map(done)