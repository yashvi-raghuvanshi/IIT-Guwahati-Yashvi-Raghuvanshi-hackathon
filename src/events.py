"""Events: give each text an event type and an impact score (1-10).

Event type:  SEC item number -> keyword rules -> zero-shot model (only if rules are unsure).
Impact:      how serious the event normally is x how strong the sentiment is x how much we
             trust the source, then ranked against RECENT news so some signals are always high.
"""
import math
import re

import numpy as np
import pandas as pd

from src import config as C

# ------------------------------------------------------------------ keyword rules
KEYWORDS = {
    "Geopolitical": ["wars?", "sanctions?", "invasion", "invad(?:e|es|ed)", "missiles?", "ceasefire", "cease-fire",
                     "tariffs?", "trade war", "tensions?", "conflict", "nato", "embargo", "airstrikes?",
                     "military", "defen[cs]e", "coup", "geopolitic\\w*", "middle east", "ukraine", "taiwan strait", "red sea", "drones?", "intercept\\w*", "attacks?", "bomb\\w*", "troops", "terror\\w*", "hostages?"],
    "Macroeconomic": ["inflation", "cpi", "ppi", "gdp", "recession", "unemployment", "jobless", "payrolls?",
                      "nonfarm", "pmi", "consumer confidence", "retail sales", "stagflation", "economic slowdown",
                      "rate hikes?", "rate cuts?", "interest rates?", "federal reserve", "(?-i:Fed)", "central bank",
                      "ecb", "rbi", "repo rate", "monetary policy", "(?:bond|treasury) yields?", "bond markets?",
                      "consumer sentiment", "rupee", "economy", "economic", "oil prices?", "crude", "brent", "diesel", "gasoline", "gas prices", "fuel prices"],
    "Credit Event": ["defaults?", "defaulted", "bankruptcy", "bankrupt", "chapter 11", "insolvency", "insolvent",
                     "credit rating", "credit downgrade", "debt downgrade", "sovereign downgrade", "junk status",
                     "debt restructuring", "bank run", "liquidity crisis", "credit crunch", "missed payment",
                     "write-?offs?", "loan losses", "non-?performing", "going concern"],
    "Merger/Acquisition": ["acquires?", "acquired", "acquiring", "acquisition", "merger", "merges?", "takeover",
                           "take over", "buyout", "agrees? to buy", "to acquire", "stake sale", "spin-?off",
                           "divest\\w*", "joint venture", "tender offer", "bid for"],
    "Product Launch": ["launch\\w*", "unveil\\w*", "introduces?", "new model", "new product", "next-gen",
                       "rollout", "rolls? out", "debut\\w*"],
    "Earnings": ["earnings", "quarterly results", "q[1-4]", "fiscal (?:year|quarter)", "revenues?", "profits?",
                 "eps", "guidance", "beats? (?:estimates|expectations)", "miss(?:es|ed)? (?:estimates|expectations)",
                 "net income", "operating income"],
    "Regulatory/Legal": ["lawsuits?", "sues?", "sued", "antitrust", "(?-i:SEC)", "probe", "investigations?",
                         "investigates?", "settlement", "regulators?", "fined", "fines", "fda", "ftc", "doj",
                         "justice department", "class action", "court", "judge", "ruling", "arrest\\w*", "fbi", "charged with", "convicted", "sentenced", "ban(?:s|ned)?", "subpoena", "indict\\w*", "fraud"],
}
_PAT = {e: re.compile(r"\b(?:" + "|".join(k) + r")\b", re.I) for e, k in KEYWORDS.items()}


def sec_event(items):
    """8-K item numbers (like '2.02') map straight to an event."""
    for it in str(items or "").split(","):
        if it.strip() in C.SEC_ITEM_TO_EVENT:
            return C.SEC_ITEM_TO_EVENT[it.strip()]
    return None


def rule_classify(text):
    """Returns (event, state, tied). state: 'clear' (one winner), 'tie' (several), 'none' (no keyword)."""
    hits = {e: len(p.findall(text)) for e, p in _PAT.items()}
    best = max(hits.values())
    if best == 0:
        return None, "none", []
    top = [e for e, h in hits.items() if h == best]
    if len(top) == 1:
        return top[0], "clear", top
    return max(top, key=lambda e: C.EVENTS[e]), "tie", top      # tie: most severe event is the fallback guess


# ------------------------------------------------------------------ zero-shot model (only for unclear texts)
ZS_LABELS = {
    "Geopolitical":       "war, sanctions or geopolitical tension",
    "Macroeconomic":      "inflation, interest rates or the economy",
    "Credit Event":       "bankruptcy, default or a credit downgrade",
    "Merger/Acquisition": "a merger, acquisition or takeover",
    "Product Launch":     "a new product launch",
    "Earnings":           "a company's quarterly earnings report or financial results",
    "Regulatory/Legal":   "a lawsuit, investigation or regulation",
    C.OTHER_EVENT:        "stock market commentary, analyst opinions, price targets, investing tips or general news",
}


class _ZeroShot:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("zero-shot-classification", model=C.ZEROSHOT_MODEL, device=-1)
        self.phrases = list(ZS_LABELS.values())
        self.to_event = {v: k for k, v in ZS_LABELS.items()}

    def scores(self, texts):
        """One {event: score} dict per text."""
        out = self.pipe([t[:300] for t in texts], candidate_labels=self.phrases,
                        hypothesis_template="This news is about {}.", multi_label=False, batch_size=8)
        if isinstance(out, dict):
            out = [out]
        return [{self.to_event[l]: s for l, s in zip(r["labels"], r["scores"])} for r in out]


_zs = None          # None = not tried yet, False = unavailable, else the model


def get_zeroshot():
    global _zs
    if _zs is None:
        try:
            _zs = _ZeroShot()
        except Exception as ex:
            print(f"[warn] zero-shot model unavailable ({type(ex).__name__}); unclear texts use the rule guess")
            _zs = False
    return _zs


def _read_zs_cache():
    if C.ZEROSHOT_CACHE.exists():
        df = pd.read_csv(C.ZEROSHOT_CACHE, dtype={"id": str})
        return dict(zip(df.id, df.event))
    return {}


def _save_zs_cache(cache):
    C.ensure_dirs()
    pd.DataFrame({"id": list(cache), "event": list(cache.values())}).to_csv(C.ZEROSHOT_CACHE, index=False)


def classify_df(df):
    """df needs 'id' and 'text' (and 'sec_items' if you have it).
    Returns a DataFrame [event, event_method] aligned with df.
    event_method: sec / rules / zeroshot / rules_tie / default (shows how each label was decided)."""
    n = len(df)
    event, method, pending = [None] * n, [""] * n, []
    sec = df["sec_items"].fillna("") if "sec_items" in df else [""] * n
    for pos, (text, items) in enumerate(zip(df.text, sec)):
        e = sec_event(items)
        if e:
            event[pos], method[pos] = e, "sec"
            continue
        top, state, tied = rule_classify(text)
        if state == "clear":
            event[pos], method[pos] = top, "rules"
        else:
            pending.append((pos, state, tied, top))

    if pending:
        ids, cache = df.id.tolist(), _read_zs_cache()
        todo = [p for p in pending if ids[p[0]] not in cache]
        zs = get_zeroshot() if todo else None
        scores = {}
        if zs:
            for p, sc in zip(todo, zs.scores([df.text.iloc[p[0]] for p in todo])):
                scores[p[0]] = sc
        changed = False
        for pos, state, tied, guess in pending:
            if ids[pos] in cache:
                event[pos], method[pos] = cache[ids[pos]], "zeroshot"
            elif pos in scores:
                sc = scores[pos]
                pool = {e: sc[e] for e in tied} if state == "tie" else sc
                best = max(pool, key=pool.get)
                event[pos] = best if (sc[best] >= C.ZS_MIN_SCORE and best != C.OTHER_EVENT) or state == "tie" else C.OTHER_EVENT
                method[pos] = "zeroshot"
                cache[ids[pos]], changed = event[pos], True
            else:                                                 # zero-shot not available
                event[pos], method[pos] = (guess, "rules_tie") if state == "tie" else (C.OTHER_EVENT, "default")
        if changed:
            _save_zs_cache(cache)
    return pd.DataFrame({"event": event, "event_method": method}, index=df.index)


# ------------------------------------------------------------------ impact score
INTENSE = ["plunge", "plummet", "crash", "collapse", "tumble", "soar", "surge", "skyrocket", "record",
           "massive", "emergency", "halt", "fraud", "crisis", "shock", "panic", "worst", "biggest"]


def raw_severity(event, sentiment, text, source):
    """Absolute severity. Symmetric: only the STRENGTH of sentiment counts, not its sign."""
    base = C.EVENTS.get(event, 3)
    boost = 0.5 * min(sum(w in text.lower() for w in INTENSE), 3)
    trust = C.SOURCES.get(source, {}).get("weight", 0.8)
    return (base * (0.55 + 0.45 * abs(sentiment)) + boost) * trust


# curve exponent chosen so that about IMPACT_TOP_SHARE of signals reach the stress trigger level
_K = math.log((C.STRESS_TRIGGER_IMPACT - 1) / 9) / math.log(1 - C.IMPACT_TOP_SHARE)


def impact_df(df):
    """df needs ts, event, sentiment, text, source. Returns impact (1-10) aligned with df.

    Each signal is ranked only against signals BEFORE it (last IMPACT_WINDOW_H hours, or the last
    IMPACT_MIN_N signals if that is more), so Replay never peeks into the future.
    The first IMPACT_MIN_N signals are ranked against the first IMPACT_MIN_N (a short warm-up)."""
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float, index=df.index)
    order = np.argsort(df.ts.values, kind="stable")
    d = df.iloc[order]
    ts = d.ts.values
    raw = np.array([raw_severity(e, s, t, src) for e, s, t, src in zip(d.event, d.sentiment, d.text, d.source)])
    left = np.searchsorted(ts, ts - np.timedelta64(C.IMPACT_WINDOW_H, "h"), side="right")
    vals = np.empty(n)
    for i in range(n):
        if i + 1 < C.IMPACT_MIN_N:
            seg = raw[:min(C.IMPACT_MIN_N, n)]
        else:
            seg = raw[min(left[i], i + 1 - C.IMPACT_MIN_N): i + 1]
        pct = ((seg < raw[i]).sum() + 0.5 * (seg == raw[i]).sum()) / len(seg)
        vals[i] = 1 + 9 * pct ** _K
    out = np.empty(n)
    out[order] = np.clip(vals, C.IMPACT_MIN, C.IMPACT_MAX)
    return pd.Series(np.round(out, 1), index=df.index)