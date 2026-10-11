"""Aggregate: turns the signal file into the numbers each dashboard panel needs.

Every function takes the signals table and an `as_of` time (None = the latest signal) and returns
plain Python data (dicts, lists, strings, numbers) that can be sent as JSON.
Nothing here looks at signals after `as_of`, so the Replay timeline never peeks into the future.
"""
import pandas as pd
import numpy as np

from src import config as C


# ------------------------------------------------------------------ small helpers
def _as_of(sig, as_of):
    if as_of is None:
        return sig.ts.max() if len(sig) else pd.Timestamp.now(tz="UTC")
    t = pd.Timestamp(as_of)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(t):
    return None if pd.isna(t) else pd.Timestamp(t).isoformat()


def _window(sig, end, hours):
    start = end - pd.Timedelta(hours=hours)
    return sig[(sig.ts > start) & (sig.ts <= end)]


def _num(x, nd=2):
    return None if pd.isna(x) else round(float(x), nd)


# ------------------------------------------------------------------ left panel + header
def timeline(sig):
    """Start and end of the replay slider."""
    if sig.empty:
        return {"start": None, "end": None}
    return {"start": _iso(sig.ts.min()), "end": _iso(sig.ts.max())}


def feed(sig, as_of=None, n=None):
    """Latest signals, newest first (the always-visible left panel)."""
    end = _as_of(sig, as_of)
    rows = sig[sig.ts <= end].sort_values("ts", ascending=False).head(n or C.FEED_SIZE)
    return [dict(id=r.signal_id, ts=_iso(r.ts), entity=r.entity, label=r.sentiment_label, score=_num(r.sentiment),
                 event=r.event, impact=_num(r.impact, 1), source=r.source, headline=r.text)
            for r in rows.itertuples()]


# ------------------------------------------------------------------ Tab 1: Risk Engine
def overview(sig, as_of=None, hours=None):
    """Total / Positive / Neutral / Negative cards, with the change in total vs the previous window."""
    end, h = _as_of(sig, as_of), hours or C.DEFAULT_WINDOW_H
    cur, prev = _window(sig, end, h), _window(sig, end - pd.Timedelta(hours=h), h)
    total = len(cur)
    out = {"window_hours": h, "total": total,
           "total_change_pct": _num((total - len(prev)) / len(prev) * 100, 1) if len(prev) else None}
    for label in ("Positive", "Neutral", "Negative"):
        n = int((cur.sentiment_label == label).sum())
        out[label.lower()] = {"count": n, "pct": _num(n / total * 100, 1) if total else 0.0}
    return out


def event_mix(sig, as_of=None, hours=None):
    """Donut chart: signals per event type."""
    cur = _window(sig, _as_of(sig, as_of), hours or C.DEFAULT_WINDOW_H)
    counts = cur.event.value_counts()
    return [{"event": e, "count": int(n), "pct": _num(n / len(cur) * 100, 1)} for e, n in counts.items()]


def sentiment_trend(sig, as_of=None, hours=None, points=12):
    """Average sentiment per bucket for Positive / Negative / Neutral signals (null where a bucket is empty)."""
    end, h = _as_of(sig, as_of), hours or C.DEFAULT_WINDOW_H
    start, step = end - pd.Timedelta(hours=h), pd.Timedelta(hours=h) / points
    cur = _window(sig, end, h)
    bucket = np.minimum(((cur.ts - start) / step).astype(float).apply(np.floor).astype(int), points - 1)
    out = {"t": [_iso(start + step * (i + 1)) for i in range(points)]}
    for name, mask in (("positive", cur.sentiment_label == "Positive"),
                       ("negative", cur.sentiment_label == "Negative"),
                       ("neutral", cur.sentiment_label == "Neutral"), ("all", cur.sentiment_label.notna())):
        m = cur.sentiment[mask].groupby(bucket[mask]).mean()
        out[name] = [_num(m.get(i), 3) for i in range(points)]
    return out


# ------------------------------------------------------------------ asset risk score (0 to 1)
def _risk_series(sig, end):
    """Risk per watchlist ticker as of `end`. Strong news in EITHER direction raises risk.
    Own-company signals count fully; market news tagged to the stock's sector counts SPILLOVER_WEIGHT."""
    s = sig[sig.ts <= end]
    age_h = (end - s.ts).dt.total_seconds() / 3600
    strength = s.sentiment.abs() * s.impact / 10 * 0.5 ** (age_h / C.RISK_HALF_LIFE_H)
    own = strength[s.entity.isin(C.WATCHLIST)].groupby(s.entity).sum()
    market = s.entity == C.MARKET_ENTITY
    by_sector = strength[market].groupby(s.sector[market]).sum()
    total = pd.Series({t: own.get(t, 0.0) + C.SPILLOVER_WEIGHT * by_sector.get(C.TICKER_TO_SECTOR[t], 0.0)
                       for t in C.WATCHLIST})
    return 1 - np.exp(-C.RISK_K * total)


def _driver(sig, end, ticker):
    """The single signal that adds most to this asset's risk."""
    sector = C.TICKER_TO_SECTOR[ticker]
    s = sig[(sig.ts <= end) & ((sig.entity == ticker) | ((sig.entity == C.MARKET_ENTITY) & (sig.sector == sector)))]
    if s.empty:
        return None
    age_h = (end - s.ts).dt.total_seconds() / 3600
    w = np.where(s.entity == ticker, 1.0, C.SPILLOVER_WEIGHT)
    top = s.loc[(s.sentiment.abs() * s.impact / 10 * 0.5 ** (age_h / C.RISK_HALF_LIFE_H) * w).idxmax()]
    return {"event": top.event, "headline": top.text, "impact": _num(top.impact, 1), "score": _num(top.sentiment)}


def asset_risk(sig, as_of=None, n=5):
    """Top assets by risk score, with the change over the last 24 hours."""
    end = _as_of(sig, as_of)
    now = _risk_series(sig, end)
    before = _risk_series(sig, end - pd.Timedelta(hours=24))
    out = []
    for rank, (tk, r) in enumerate(now.sort_values(ascending=False).head(n).items(), 1):
        out.append(dict(rank=rank, ticker=tk, sector=C.TICKER_TO_SECTOR[tk], risk=_num(r), change_24h=_num(r - before[tk]),
                        driver=_driver(sig, end, tk)))
    return out


# ------------------------------------------------------------------ Tab 3 banner
def trigger_status(sig, as_of=None, hours=None):
    """HIGH RISK / ELEVATED / NORMAL, plus the signals that reached the stress-test level."""
    cur = _window(sig, _as_of(sig, as_of), hours or C.DEFAULT_WINDOW_H)
    hits = cur[cur.impact >= C.STRESS_TRIGGER_IMPACT].sort_values("impact", ascending=False)
    top = cur.impact.max() if len(cur) else 0
    status = "HIGH RISK" if top >= C.HIGH_RISK_IMPACT else "ELEVATED" if top >= C.STRESS_TRIGGER_IMPACT else "NORMAL"
    return {"status": status, "max_impact": _num(top, 1), "trigger_count": len(hits),
            "triggers": [dict(id=r.signal_id, ts=_iso(r.ts), entity=r.entity, event=r.event, impact=_num(r.impact, 1),
                              score=_num(r.sentiment), headline=r.text) for r in hits.head(10).itertuples()]}


# ------------------------------------------------------------------ one call for a whole screen refresh
def snapshot(sig, as_of=None, hours=None):
    """Everything the dashboard needs for one moment in time, in a single call (keeps loading fast)."""
    end = _as_of(sig, as_of)
    return {"as_of": _iso(end), "timeline": timeline(sig), "feed": feed(sig, end),
            "overview": overview(sig, end, hours), "events": event_mix(sig, end, hours),
            "trend": sentiment_trend(sig, end, hours), "top_risk": asset_risk(sig, end),
            "trigger": trigger_status(sig, end, hours)}