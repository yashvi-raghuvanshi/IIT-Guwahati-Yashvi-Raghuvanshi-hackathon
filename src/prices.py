"""Prices: downloads market prices for the watchlist and keeps a saved copy for offline use.

Daily closes by default (set PRICE_INTERVAL = "1h" in config.py for hourly bars).
A daily close is stamped 21:00 UTC, the time it becomes known, so weights decided at time t
are only ever compared with returns that happen AFTER t.
If prices cannot be loaded, functions return None and the dashboard hides the performance panel.
"""
import pandas as pd

from src import config as C

YAHOO_TO_OURS = {c["yahoo"]: t for t, c in C.WATCHLIST.items()}      # BRK-B -> BRK.B


def _path(interval=None):
    return C.PROCESSED / f"prices_{interval or C.PRICE_INTERVAL}.csv"


def _normalise_index(idx, interval):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    if interval.endswith("d"):                                       # daily bar: known only after the close
        return (idx.normalize() + pd.Timedelta(hours=C.CLOSE_HOUR_UTC)).tz_localize("UTC")
    return idx.tz_localize("UTC")


# ------------------------------------------------------------------ download + cache
def fetch_prices(period=None, interval=None):
    """Download from Yahoo Finance. Returns a table: rows = time (UTC), columns = our tickers."""
    import yfinance as yf
    period, interval = period or C.PRICE_PERIOD, interval or C.PRICE_INTERVAL
    raw = yf.download(list(YAHOO_TO_OURS), period=period, interval=interval, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo returned no prices")
    close = raw["Close"].rename(columns=YAHOO_TO_OURS)
    close.index = _normalise_index(close.index, interval)
    close = close.dropna(how="all").dropna(axis=1, how="all").sort_index()
    if close.empty:
        raise RuntimeError("Yahoo returned no prices")
    return close


def _save(df):
    C.ensure_dirs()
    old = load_prices()
    if old is not None:
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()      # new numbers win over old ones
    df.rename_axis("ts").to_csv(_path())
    return df


def load_prices():
    """The saved copy, or None if there is none."""
    p = _path()
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col="ts", parse_dates=["ts"])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC") if df.index.tz else pd.DatetimeIndex(df.index).tz_localize("UTC")
    return df


def refresh():
    """Download fresh prices and merge them into the saved copy. None (with a warning) if that fails."""
    try:
        return _save(fetch_prices())
    except Exception as ex:
        print(f"[warn] price download failed ({type(ex).__name__}: {ex}); using the saved copy if there is one")
        return load_prices()


def get_prices(refresh_first=False):
    """Prices for Module A. None means: hide the performance panel."""
    return refresh() if refresh_first else load_prices()


# ------------------------------------------------------------------ helpers for the rebalancer / dashboard
def status(prices=None):
    """Tells the dashboard whether the performance panel can be shown."""
    p = load_prices() if prices is None else prices
    if p is None or p.empty:
        return {"available": False, "interval": C.PRICE_INTERVAL, "start": None, "end": None, "tickers": 0, "missing": list(C.WATCHLIST)}
    return {"available": True, "interval": C.PRICE_INTERVAL, "start": p.index.min().isoformat(), "end": p.index.max().isoformat(),
            "tickers": int(p.shape[1]), "missing": [t for t in C.WATCHLIST if t not in p.columns]}


def as_of(prices, when):
    """Only the bars known at time `when`."""
    t = pd.Timestamp(when)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return prices[prices.index <= t]


def bar_returns(prices):
    """Return from one bar to the next. Short gaps (a missing day for one stock) are filled forward."""
    return prices.ffill(limit=3).pct_change()


if __name__ == "__main__":
    df = refresh()
    print(status(df))