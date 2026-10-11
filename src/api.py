"""API: gives the dashboard everything it needs, and refreshes the data in the background.

Run:  python -m src.api        then open http://localhost:8000   (API docs at /docs)

The server keeps no clock. Every call can carry an `as_of` time (ISO, e.g. 2026-10-09T12:00:00Z);
leave it out to get the latest. Replay = the dashboard sends times, Live = it sends none.
"""
import json
import math
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import aggregate as A
from src import config as C
from src import engine, portfolio
from src import prices as P
from src import rebalancer as R
from src import stress as S


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(x):
    """Make anything JSON-safe: numpy numbers become plain numbers, NaN/inf become null."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _empty_signals():
    df = pd.DataFrame(columns=C.SIGNAL_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# ------------------------------------------------------------------ data held in memory
class Store:
    """Signals, the Module A weights path, prices and the portfolio, loaded once and swapped as a whole."""

    def __init__(self):
        sig = _empty_signals()
        self._state = (sig, R.rebalance_path(sig), None, portfolio.load_portfolio())
        self.mtime, self.version, self.loaded_at, self.load_error = None, 0, None, None
        self._lock = threading.Lock()
        self.refresh = dict(enabled=C.AUTO_REFRESH, every_minutes=C.REFRESH_MINUTES, running=False, runs=0,
                            last_start=None, last_end=None, last_error=None, next_run=None)

    def state(self):
        return self._state                                    # (signals, weights path, prices, portfolio)

    @staticmethod
    def _file():
        for f in (C.SIGNALS_PARQUET, C.SIGNALS_CSV):
            if f.exists():
                return f
        return None

    def reload(self):
        with self._lock:
            f = self._file()
            sig = engine.load_signals().sort_values("ts").reset_index(drop=True) if f else _empty_signals()
            new = (sig, R.rebalance_path(sig), P.get_prices(), portfolio.load_portfolio())
            self._state = new                                 # one assignment: readers never see half old, half new
            self.mtime = f.stat().st_mtime if f else None
            self.version += 1
            self.loaded_at, self.load_error = _now(), None

    def maybe_reload(self):
        """Pick up a new signals file written from outside (for example you ran the engine by hand)."""
        if self.refresh["running"]:
            return                                            # the refresher reloads when it is done
        f = self._file()
        m = f.stat().st_mtime if f else None
        if m != self.mtime and (m is None or time.time() - m > 2):     # wait until the file stopped changing
            try:
                self.reload()
            except Exception as ex:
                self.mtime, self.load_error = m, f"{type(ex).__name__}: {ex}"   # keep the old data, do not retry in a loop


store = Store()
_refresh_lock = threading.Lock()


# ------------------------------------------------------------------ background refresh (Live mode)
def refresh_once():
    """Fetch fresh data, run the engine, refresh prices, reload. Errors are recorded, never raised."""
    if not _refresh_lock.acquire(blocking=False):
        return
    st = store.refresh
    st.update(running=True, last_start=_now())
    try:
        engine.run("live")
        P.refresh()
        store.reload()
        st["last_error"] = None
    except Exception as ex:
        st["last_error"] = f"{type(ex).__name__}: {ex}"
    finally:
        st.update(running=False, last_end=_now(), runs=st["runs"] + 1)
        _refresh_lock.release()


def _refresh_loop(stop, first_wait=10.0):
    wait = first_wait
    while True:
        store.refresh["next_run"] = (datetime.now(timezone.utc) + timedelta(seconds=wait)).isoformat(timespec="seconds")
        if stop.wait(wait):
            return
        refresh_once()
        wait = C.REFRESH_MINUTES * 60


@asynccontextmanager
async def lifespan(app):
    store.reload()
    stop = threading.Event()
    if C.AUTO_REFRESH:
        threading.Thread(target=_refresh_loop, args=(stop,), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="NLP Risk Engine API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ input checks
def _t(as_of):
    if not as_of:
        return None
    try:
        if "T" in as_of:
            as_of = as_of.replace(" ", "+")                   # in an ISO time, a space is a '+' that the URL turned into a space
        t = pd.Timestamp(as_of)
    except Exception:
        raise HTTPException(400, f"as_of must be an ISO time like 2026-10-09T12:00:00Z, got {as_of!r}")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _hours(hours):
    if hours is not None and not (0 < hours <= 168):
        raise HTTPException(400, "hours must be between 0 and 168")
    return hours


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health():
    store.maybe_reload()
    sig, _, prices, _ = store.state()
    try:
        meta = json.loads(C.ENGINE_META.read_text())
    except Exception:
        meta = {}
    return _clean({"status": "ok" if len(sig) else "no data: run  python -m src.engine live",
                   "signals": len(sig), "default_mode": C.MODE, "sentiment_model": meta.get("sentiment_model"),
                   "engine_last_run": meta.get("generated_at"), "loaded_at": store.loaded_at, "data_version": store.version,
                   "load_error": store.load_error, "prices": P.status(prices), "refresh": store.refresh})


@app.get("/api/timeline")
def timeline():
    store.maybe_reload()
    tl = A.timeline(store.state()[0])
    return _clean({**tl, "default_as_of": tl["end"], "speeds": C.REPLAY_SPEEDS,
                   "default_mode": C.MODE, "window_days": C.REPLAY_DAYS})


@app.get("/api/overview")
def overview(as_of: Optional[str] = None, hours: Optional[float] = None):
    """Left feed + the Risk Engine tab + the stress banner, in one call."""
    store.maybe_reload()
    return _clean(A.snapshot(store.state()[0], _t(as_of), _hours(hours)))


@app.get("/api/rebalance")
def rebalance(as_of: Optional[str] = None, top: int = 5):
    store.maybe_reload()
    sig, path, prices, _ = store.state()
    return _clean(R.module_a(sig, _t(as_of), top=max(1, min(top, 15)), prices=prices, path=path))


@app.get("/api/stress/auto")
def stress_auto(as_of: Optional[str] = None, hours: Optional[float] = None):
    """The banner, plus the automatic stress test when an event reached the trigger level."""
    store.maybe_reload()
    sig, _, _, book = store.state()
    t, h = _t(as_of), _hours(hours)
    result = S.triggered_stress(sig, book, t, h)
    return _clean({"triggered": result is not None, "banner": A.trigger_status(sig, t, h), "result": result})


@app.get("/api/stress/scenarios")
def stress_scenarios():
    return {"scenarios": S.scenario_list(),
            "ranges": {"rate_bps": [-300, 500], "oil_pct": [-60, 100], "vix_pct": [-30, 150], "gdp_pts": [-6, 2]}}


class StressRequest(BaseModel):
    scenario: Optional[str] = None             # a button name, or leave out and send the four slider values
    rate_bps: float = Field(0, ge=-1000, le=1000)
    oil_pct: float = Field(0, ge=-90, le=300)
    vix_pct: float = Field(0, ge=-90, le=500)
    gdp_pts: float = Field(0, ge=-15, le=10)


@app.post("/api/stress/run")
def stress_run(req: StressRequest):
    """The 'Fire Scenario' button."""
    store.maybe_reload()
    book = store.state()[3]
    if req.scenario:
        if req.scenario not in S.SCENARIOS:
            raise HTTPException(404, f"Unknown scenario {req.scenario!r}. Choose from {list(S.SCENARIOS)}")
        return _clean(S.run_scenario(book, req.scenario))
    return _clean(S.run_custom(book, req.rate_bps, req.oil_pct, req.vix_pct, req.gdp_pts))


@app.get("/api/portfolio")
def portfolio_info():
    book = store.state()[3]
    top = book.head(10)[["asset_id", "counterparty", "type", "subtype", "sector", "rating", "market_value", "weight"]]
    return _clean({"summary": portfolio.summary(book), "top_positions": top.to_dict("records")})


# ------------------------------------------------------------------ the dashboard files (added last so /api routes win)
if C.WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(C.WEB_DIR), html=True), name="web")
else:
    @app.get("/")
    def root():
        return {"message": "API is running. Put the dashboard files in the web/ folder to see them here.",
                "docs": "/docs", "health": "/api/health"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=C.API_HOST, port=C.API_PORT)