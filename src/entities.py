"""Entities: who is a text about, and which sector does it belong to?

Order of checks:
  1. Watchlist names (apple, tesla...), cashtags ($TSLA) and exchange tags (NASDAQ: AAPL).
  2. spaCy (if installed) confirms tricky names (Apple the company, not the fruit) and spots
     spelling variants the plain search misses (J.P. Morgan).
  3. If several companies appear, the first one mentioned wins.
  4. Nothing found -> MARKET, tagged to a sector when sector words appear (oil -> Energy).
Companies that are not on the watchlist are ignored on purpose.
"""
import re

import pandas as pd

from src import config as C

AMBIGUOUS = {"apple", "amazon", "azure", "berkshire"}     # also mean a fruit, a river, a colour, a county
HINT_TRUSTED_SOURCES = {"sec_8k", "tweets"}               # these sources already know the company

# ------------------------------------------------------------------ plain-text matching
_ALIASES = sorted(C.ALIAS_TO_TICKER, key=len, reverse=True)           # longest first
_ALIAS_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ALIASES) + r")\b", re.I)
_TICKERS_NORM = {t.replace(".", "").replace("-", ""): t for t in C.WATCHLIST}   # BRKB -> BRK.B
_CASHTAG = re.compile(r"\$([A-Za-z]{1,5})(?:[.\-]([A-Za-z]))?\b")
_EXCHANGE = re.compile(r"\b(?:NYSE|NASDAQ)\s*:\s*([A-Z]{1,5})(?:[.\-]([A-Z]))?\b")


def _regex_candidates(text):
    """List of (start, end, ticker, needs_confirmation)."""
    found = [(m.start(), m.end(), C.ALIAS_TO_TICKER[m.group(1).lower()], m.group(1).lower() in AMBIGUOUS)
             for m in _ALIAS_RE.finditer(text)]
    for rx in (_CASHTAG, _EXCHANGE):
        for m in rx.finditer(text):
            tk = _TICKERS_NORM.get((m.group(1) + (m.group(2) or "")).upper())
            if tk:
                found.append((m.start(), m.end(), tk, False))
    return found


# ------------------------------------------------------------------ spaCy (optional)
_nlp = None            # None = not tried yet, False = unavailable


def _get_ner():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            nlp = spacy.load(C.SPACY_MODEL)
            nlp.select_pipes(enable="ner")                # switch off everything except entity finding (faster)
            _nlp = nlp
        except Exception as ex:
            print(f"[warn] spaCy unavailable ({type(ex).__name__}); skipping name confirmation")
            _nlp = False
    return _nlp


def _orgs(texts):
    """Company-like spans per text as (start, end, text). None if spaCy is not available."""
    nlp = _get_ner()
    if not nlp:
        return None
    return [[(e.start_char, e.end_char, e.text) for e in doc.ents if e.label_ in C.NER_LABELS]
            for doc in nlp.pipe(texts, batch_size=64)]


def _norm_org(name):
    """'J.P. Morgan Chase & Co.' -> 'jp morgan chase &'"""
    s = name.lower().replace(".", "")
    s = re.sub(r"\b(inc|corp|corporation|co|company|ltd|plc|llc|holdings|group|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _org_to_ticker(norm):
    for a in _ALIASES:
        if re.search(rf"\b{re.escape(a)}\b", norm):
            return C.ALIAS_TO_TICKER[a]
    return None


# ------------------------------------------------------------------ sector words for market-wide news
SECTOR_WORDS = {
    "Energy":     ["oil", "crude", "opec", "petroleum", "refiner\\w*", "natural gas", "brent", "wti", "lng", "pipelines?", "energy"],
    "Financials": ["(?<!central )banks?", "banking", "lenders?", "lending", "insurers?", "insurance", "mortgages?", "brokerages?", "credit cards?"],
    "Technology": ["chips?", "semiconductors?", "software", "cloud", "data cent(?:er|re)s?", "artificial intelligence", "(?-i:AI)", "tech"],
    "Healthcare": ["pharma\\w*", "drugmakers?", "biotech\\w*", "vaccines?", "hospitals?", "healthcare", "health care", "clinical trials?", "medicare"],
    "Consumer":   ["retail\\w*", "consumers?", "consumer spending", "e-commerce", "grocer\\w*", "restaurants?", "shoppers?"],
}
_SECTOR_RE = {s: re.compile(r"\b(?:" + "|".join(w) + r")\b", re.I) for s, w in SECTOR_WORDS.items()}


def sector_from_words(text):
    hits = {s: len(p.findall(text)) for s, p in _SECTOR_RE.items()}
    best = max(hits, key=hits.get)                     # ties go to the first sector listed
    return best if hits[best] else "Others"


# ------------------------------------------------------------------ main function
def link_df(df):
    """df needs 'text'; uses 'source' and 'ticker_hint' when present.
    Returns a DataFrame [entity, entity_type, sector] aligned with df."""
    texts = df.text.tolist()
    sources = df["source"].tolist() if "source" in df else [""] * len(texts)
    hints = df["ticker_hint"].tolist() if "ticker_hint" in df else [None] * len(texts)

    cands = [_regex_candidates(t) for t in texts]
    need = [i for i, c in enumerate(cands) if not c or any(x[3] for x in c)]   # spaCy only where it can help
    spans = dict(zip(need, _orgs([texts[i] for i in need]) or [])) if need else {}

    rows = []
    for i, text in enumerate(texts):
        orgs = spans.get(i)                            # None when spaCy did not look at this text
        valid = []
        for start, end, tk, needs_check in cands[i]:
            if needs_check and orgs is not None and not any(s < end and start < e for s, e, _ in orgs):
                continue                               # "Amazon rainforest": not a company
            valid.append((start, tk))
        for s, _, name in (orgs or []):                # spelling variants the plain search missed
            tk = _org_to_ticker(_norm_org(name))
            if tk and tk not in {v[1] for v in valid}:
                valid.append((s, tk))
        tickers = [tk for _, tk in sorted(valid)]

        hint = hints[i] if isinstance(hints[i], str) and hints[i] in C.WATCHLIST else None
        if hint and (sources[i] in HINT_TRUSTED_SOURCES or hint in tickers):
            entity = hint
        else:
            entity = tickers[0] if tickers else None

        if entity:
            rows.append((entity, "watchlist", C.TICKER_TO_SECTOR[entity]))
        else:
            rows.append((C.MARKET_ENTITY, "market", sector_from_words(text)))
    return pd.DataFrame(rows, columns=["entity", "entity_type", "sector"], index=df.index)