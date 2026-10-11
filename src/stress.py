"""Stress: Module B. Turns an event (or a manual scenario) into shocks and applies them to the portfolio.

A shock is four numbers (plus optional sector hits):
    dy   change in yields          (+200 bps = +0.02)
    oil  change in the oil price   (-30%     = -0.30)
    vix  change in the VIX         (+50%     = +0.50)
    gdp  change in GDP growth      (-2 points = -2.0)
Each position's change in value = -rate_dur*dy + oil_beta*oil + vix_beta*vix + gdp_beta*gdp (+ sector hit).
"""
import numpy as np
import pandas as pd

from src import config as C
from src.aggregate import _as_of, _window, _iso

SECTOR_PASS = {"Equity": 1.0, "Bond": 0.25, "Loan": 0.20, "Derivative": 0.30}   # how much of a sector hit each asset type feels


def shock(rate_bps=0, oil_pct=0, vix_pct=0, gdp_pts=0, sector=None):
    """Build a shock from dashboard-style numbers (bps, %, %, points)."""
    return dict(dy=rate_bps / 1e4, oil=oil_pct / 100, vix=vix_pct / 100, gdp=float(gdp_pts), sector=dict(sector or {}))


def scale_shock(s, k):
    return dict(dy=s["dy"] * k, oil=s["oil"] * k, vix=s["vix"] * k, gdp=s["gdp"] * k,
                sector={a: b * k for a, b in s["sector"].items()})


def sliders(s):
    """A shock back in dashboard units, to move the sliders when a scenario is picked."""
    return dict(rate_bps=round(s["dy"] * 1e4), oil_pct=round(s["oil"] * 100), vix_pct=round(s["vix"] * 100),
                gdp_pts=round(s["gdp"], 1))


# ------------------------------------------------------------------ what each event does to the world (full shock = impact 7)
EVENT_SHOCKS = {
    "Geopolitical":       dict(rate_bps=-25, oil_pct=25, vix_pct=60, gdp_pts=-1.5),
    "Macroeconomic":      dict(rate_bps=50, vix_pct=25, gdp_pts=-1.0),
    "Credit Event":       dict(rate_bps=-50, vix_pct=50, gdp_pts=-1.5, sector={"Financials": -0.06}),
    "Regulatory/Legal":   dict(vix_pct=15, gdp_pts=-0.3),
    "Earnings":           dict(vix_pct=10, gdp_pts=-0.2),            # company news: small effect on a sector-level book
    "Merger/Acquisition": dict(vix_pct=5),
    "Product Launch":     dict(vix_pct=3),
}

# the four buttons in the dashboard
SCENARIOS = {
    "Global Recession": dict(rate_bps=-100, oil_pct=-25, vix_pct=80, gdp_pts=-2.0),
    "Oil Price Surge":  dict(rate_bps=50, oil_pct=40, vix_pct=30, gdp_pts=-1.0),
    "Rate Hike":        dict(rate_bps=200, vix_pct=20, gdp_pts=-0.5),
    "Tech Selloff":     dict(vix_pct=50, gdp_pts=-0.5, sector={"Technology": -0.12}),
}


def scenario_list():
    return [{"name": n, "sliders": sliders(shock(**kw))} for n, kw in SCENARIOS.items()]


# ------------------------------------------------------------------ applying a shock
def apply_shock(book, s):
    """Per-position results. Returns the book with pnl, pct and the contribution of each factor."""
    d = book.copy()
    comp = {"rates": -d.rate_dur * s["dy"], "oil": d.oil_beta * s["oil"], "vix": d.vix_beta * s["vix"],
            "gdp": d.gdp_beta * s["gdp"],
            "sector": d.sector.map(s["sector"]).fillna(0.0) * d.type.map(SECTOR_PASS)}
    dv = np.clip(sum(comp.values()), -1.0, None)               # a position cannot lose more than all of its value
    d["pct"] = dv
    d["pnl"] = d.market_value * dv
    d["value_after"] = d.market_value + d.pnl
    for k, v in comp.items():
        d[f"pnl_{k}"] = d.market_value * v
    return d


# ------------------------------------------------------------------ 30-day path (before vs after the shock)
PATH_DAYS, SHOCK_DAY, RAMP_DAYS = 30, 5, 4


def value_path(before_value, shock_pct):
    """Daily portfolio value without and with the shock. The shock starts on SHOCK_DAY, builds over RAMP_DAYS, then stays.
    The 'before' line uses a fixed random seed, so it does not jitter when the scenario changes."""
    rng = np.random.default_rng(C.PORTFOLIO_SEED + 1)
    base = before_value * np.concatenate([[1.0], np.cumprod(1 + rng.normal(0.0002, 0.0035, PATH_DAYS))])
    ramp = np.clip((np.arange(PATH_DAYS + 1) - SHOCK_DAY) / RAMP_DAYS, 0, 1)
    after = base * (1 + shock_pct * (1 - (1 - ramp) ** 2))     # ease-out: fast at first, then settles
    return {"day": list(range(PATH_DAYS + 1)), "shock_day": SHOCK_DAY,
            "before": [round(float(v)) for v in base], "after": [round(float(v)) for v in after],
            "before_idx": [round(float(v / base[0] * 100), 2) for v in base],
            "after_idx": [round(float(v / base[0] * 100), 2) for v in after]}


def run_stress(book, s, label="Custom scenario"):
    d = apply_shock(book, s)
    before, pnl = float(d.market_value.sum()), float(d.pnl.sum())

    def table(col, keys):
        g = d.groupby(col).agg(before=("market_value", "sum"), pnl=("pnl", "sum")).reindex(keys, fill_value=0.0)
        return [dict(name=k, before=round(float(r.before)), after=round(float(r.before + r.pnl)), pnl=round(float(r.pnl)),
                     pnl_pct=round(float(r.pnl / r.before * 100), 2) if r.before else 0.0) for k, r in g.iterrows()]

    by_sector = sorted(table("sector", C.SECTORS), key=lambda r: r["pnl"])
    losers = d.nsmallest(10, "pnl")
    return {"label": label, "shock": sliders(s), "sector_hits": s["sector"],
            "before": round(before), "after": round(before + pnl), "pnl": round(pnl), "pnl_pct": round(pnl / before * 100, 2),
            "by_sector": by_sector, "by_type": table("type", ["Loan", "Bond", "Derivative", "Equity"]),
            "by_factor": {k: round(float(d[f"pnl_{k}"].sum())) for k in ("rates", "oil", "vix", "gdp", "sector")},
            "top_losers": [dict(asset_id=r.asset_id, counterparty=r.counterparty, type=r.type, sector=r.sector,
                                pnl=round(float(r.pnl)), pct=round(float(r.pct * 100), 2)) for r in losers.itertuples()],
            "path": value_path(before, pnl / before)}


# ------------------------------------------------------------------ the three ways to start a test
def run_scenario(book, name):
    return run_stress(book, shock(**SCENARIOS[name]), label=name)


def run_custom(book, rate_bps=0, oil_pct=0, vix_pct=0, gdp_pts=0):
    """The 'Fire Scenario' button: whatever the four sliders say."""
    return run_stress(book, shock(rate_bps, oil_pct, vix_pct, gdp_pts), label="Custom scenario")


def triggered_stress(sig, book, as_of=None, hours=None):
    """Automatic test. Every event in the window with impact >= trigger level (direction ignored) is a candidate.
    Shock = event bundle x (impact / trigger level), so impact 7 = full shock and 10 = about 1.4x.
    The candidate that hurts the portfolio most is the one we show (ties: higher impact). None if nothing fires."""
    end = _as_of(sig, as_of)
    cur = _window(sig, end, hours or C.DEFAULT_WINDOW_H)
    cand = cur[(cur.impact >= C.STRESS_TRIGGER_IMPACT) & cur.event.isin(EVENT_SHOCKS)]
    if cand.empty:
        return None
    best = None
    for row in cand.itertuples():
        s = scale_shock(shock(**EVENT_SHOCKS[row.event]), float(row.impact) / C.STRESS_TRIGGER_IMPACT)
        loss = float(apply_shock(book, s).pnl.sum())
        if best is None or (loss, -row.impact) < (best[0], -best[1].impact):
            best = (loss, row, s)
    _, top, s = best
    res = run_stress(book, s, label=f"{top.event}: {top.text}")
    res["trigger"] = dict(id=top.signal_id, ts=_iso(top.ts), entity=top.entity, event=top.event, impact=round(float(top.impact), 1),
                          sentiment=round(float(top.sentiment), 2), shock_scale=round(float(top.impact) / C.STRESS_TRIGGER_IMPACT, 2),
                          headline=top.text, candidates=int(len(cand)))
    return res