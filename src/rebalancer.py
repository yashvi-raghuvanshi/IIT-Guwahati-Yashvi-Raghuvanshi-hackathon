"""Rebalancer: Module A. Moves the weights of a 15-stock index with company sentiment, once an hour.

The rule (linear tilt), every hour:
    score    = recent sentiment of the company's news, from -1 to +1 (newer and higher-impact news counts more)
    target   = start weight x (1 + score)          e.g. start 6.7%, score +0.3 -> about 8.7%
    then:      keep every stock between 2% and 12%, and trade at most 10% of the portfolio per step
Only company news is used (market-wide news does not move weights). Nothing here looks past the current time.
"""
import numpy as np
import pandas as pd

from src import config as C
from src import prices as P
from src.aggregate import _as_of, _iso

TICKERS = list(C.WATCHLIST)
N = len(TICKERS)
BASE = np.full(N, 1.0 / N)                         # starting weights: equal, the same as the benchmark
PATH_COLS = ["ts", "ticker", "weight", "target", "score", "turnover"]


# ------------------------------------------------------------------ the weight rule
def _bounded(w, lo, hi):
    """Make weights add up to 1 with every weight between lo and hi."""
    c = np.clip(w / w.sum(), lo, hi)
    for _ in range(100):
        diff = 1 - c.sum()
        if abs(diff) < 1e-12:
            break
        room = (c < hi - 1e-12) if diff > 0 else (c > lo + 1e-12)
        if not room.any():
            break
        c[room] += diff * c[room] / c[room].sum()
        c = np.clip(c, lo, hi)
    return c


def _scores(own, t):
    """Sentiment score per stock at time t, from company news up to t only."""
    past = own[own.ts <= t]
    if past.empty:
        return np.zeros(N)
    age_h = (t - past.ts).dt.total_seconds() / 3600
    w = 0.5 ** (age_h / C.REBAL_HALF_LIFE_H) * past.impact / 10
    num = (w * past.sentiment).groupby(past.entity).sum()
    den = w.groupby(past.entity).sum()
    return (num / (den + C.REBAL_SHRINK)).reindex(TICKERS).fillna(0.0).clip(-1, 1).values


def rebalance_path(sig):
    """Weights at every hourly step. Returns a long table: ts, ticker, weight, target, score, turnover."""
    if sig.empty:
        return pd.DataFrame(columns=PATH_COLS)
    own = sig[sig.entity.isin(C.WATCHLIST)]
    steps = pd.date_range(sig.ts.min().floor("h"), sig.ts.max().ceil("h"), freq=f"{C.REBAL_FREQ_H}h")
    prev, rows = BASE.copy(), []
    for t in steps:
        score = _scores(own, t)
        target = _bounded(BASE * (1 + C.REBAL_TILT * score), C.REBAL_MIN_W, C.REBAL_MAX_W)
        wanted = 0.5 * np.abs(target - prev).sum()                         # share of the portfolio we would like to trade
        step = 1.0 if wanted <= C.REBAL_TURNOVER_CAP else C.REBAL_TURNOVER_CAP / wanted
        w = prev + step * (target - prev)                                   # move only part of the way if over the cap
        turnover = 0.5 * np.abs(w - prev).sum()
        rows += [(t, tk, w[i], target[i], score[i], turnover) for i, tk in enumerate(TICKERS)]
        prev = w
    return pd.DataFrame(rows, columns=PATH_COLS)


# ------------------------------------------------------------------ reading the path
def _wide(path, col="weight"):
    if path.empty:                                   # no signals yet: an empty table with the right columns
        return pd.DataFrame(columns=TICKERS, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    return path.pivot(index="ts", columns="ticker", values=col)[TICKERS]


def weights_at(path, when):
    """Weights in force at a time (equal weights before the first step)."""
    if path.empty:
        return pd.Series(BASE, index=TICKERS)
    W = _wide(path)
    pos = W.index.searchsorted(pd.Timestamp(when), side="right") - 1
    return pd.Series(BASE, index=TICKERS) if pos < 0 else W.iloc[pos]


def weights_series(path, as_of, top=5):
    """'Index Weights Over Time': the biggest stocks right now, the rest added up as Others."""
    W = _wide(path)
    W = W[W.index <= as_of]
    if W.empty:
        return {"t": [], "weights": {}}
    keep = W.iloc[-1].nlargest(top).index
    out = {tk: [round(float(v), 4) for v in W[tk]] for tk in keep}
    out["Others"] = [round(float(v), 4) for v in W.drop(columns=keep).sum(axis=1)]
    return {"t": [_iso(t) for t in W.index], "weights": out}


def weights_table(path, as_of):
    """'Current vs Starting Weights', biggest change first."""
    now = weights_at(path, as_of)
    score = _wide(path, "score")
    score = score[score.index <= as_of]
    rows = [dict(ticker=tk, sector=C.TICKER_TO_SECTOR[tk], start=round(float(BASE[i]), 4), current=round(float(now[tk]), 4),
                 change_pp=round(float(now[tk] - BASE[i]) * 100, 2),
                 score=round(float(score.iloc[-1][tk]), 2) if len(score) else 0.0)
            for i, tk in enumerate(TICKERS)]
    return sorted(rows, key=lambda r: -abs(r["change_pp"]))


# ------------------------------------------------------------------ performance vs equal weight (needs prices)
def performance(path, prices, as_of):
    """Dynamic index vs the equal-weight benchmark, from daily bars. {'available': False} hides the panel."""
    if prices is None or prices.empty or path.empty:
        return {"available": False}
    p = P.as_of(prices, as_of)
    start = max(p.index.searchsorted(path.ts.min(), side="right") - 1, 0)   # last close at or before the first step
    p = p.iloc[start:]
    if len(p) < 2:
        return {"available": True, "obs": 0, "dates": [], "dynamic": [], "benchmark": []}
    rets = P.bar_returns(p).iloc[1:]
    dyn, bench = [], []
    for k in range(len(rets)):
        w = weights_at(path, p.index[k])                    # weights known at the PREVIOUS close, applied to the next return
        r = rets.iloc[k]
        ok = r.notna() & w.reindex(r.index).notna()
        wk = w.reindex(r.index)[ok]
        dyn.append(float((wk / wk.sum() * r[ok]).sum()))
        bench.append(float(r[ok].mean()))
    dyn_idx, bench_idx = 100 * np.cumprod(1 + np.array(dyn)), 100 * np.cumprod(1 + np.array(bench))
    active = np.array(dyn) - np.array(bench)
    return {"available": True, "obs": len(dyn), "interval": C.PRICE_INTERVAL,
            "dates": [_iso(p.index[0])] + [_iso(t) for t in rets.index],
            "dynamic": [100.0] + [round(float(v), 3) for v in dyn_idx], "benchmark": [100.0] + [round(float(v), 3) for v in bench_idx],
            "total_return_pct": round(float(dyn_idx[-1] - 100), 2), "benchmark_return_pct": round(float(bench_idx[-1] - 100), 2),
            "excess_pct": round(float(dyn_idx[-1] - bench_idx[-1]), 2),
            "tracking_error_pct": round(float(active.std(ddof=1) * 100), 2) if len(active) >= 3 else None}


# ------------------------------------------------------------------ one call for the whole tab
def module_a(sig, as_of=None, top=5, prices="load", path=None):
    """Everything the Index Rebalancing tab needs. Pass `path` (from rebalance_path) to avoid recomputing it."""
    end = _as_of(sig, as_of)
    path = rebalance_path(sig) if path is None else path
    if isinstance(prices, str):
        prices = P.get_prices()
    done = path[path.ts <= end]
    return {"as_of": _iso(end), "weights_series": weights_series(path, end, top), "table": weights_table(path, end),
            "performance": performance(path, prices, end),
            "rules": {"rebalance_every_hours": C.REBAL_FREQ_H, "min_weight": C.REBAL_MIN_W, "max_weight": C.REBAL_MAX_W,
                      "turnover_cap": C.REBAL_TURNOVER_CAP, "rebalances": int(done.ts.nunique()),
                      "last_turnover_pct": round(float(done.turnover.iloc[-1]) * 100, 2) if len(done) else 0.0}}