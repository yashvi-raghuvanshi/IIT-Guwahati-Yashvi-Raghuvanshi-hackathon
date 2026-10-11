"""Portfolio: builds the synthetic wholesale-banking book that Module B stress-tests.

Everything is made up (fictional counterparties, fixed random seed). Only the SHAPE of real data is borrowed:
  - sizes are heavy-tailed (a few big exposures, many small ones), like transaction amounts in the sample datasets
  - the sector mix follows a spending-category style split, mapped to our six sectors
  - the same counterparty appears several times (loan + bond + swap), so risk is concentrated
Every position carries sensitivities the stress model multiplies by a shock:
    rate_dur : change in value = -rate_dur x (change in yield, e.g. +200bps = +0.02)
    oil_beta : change in value =  oil_beta x (change in oil price, e.g. -30% = -0.30)
    vix_beta : change in value =  vix_beta x (change in VIX, e.g. +50% = +0.50)
    gdp_beta : change in value =  gdp_beta x (change in GDP growth in points, e.g. -2 points = -2)
"""
import numpy as np
import pandas as pd

from src import config as C

TYPES = ["Loan", "Bond", "Derivative", "Equity"]
COUNT_SHARE = dict(zip(TYPES, [0.38, 0.26, 0.18, 0.18]))     # share of positions
VALUE_SHARE = dict(zip(TYPES, [0.40, 0.30, 0.15, 0.15]))     # share of total value
SECTOR_SHARE = dict(zip(C.SECTORS, [0.22, 0.24, 0.20, 0.12, 0.12, 0.10]))
MAX_POSITION = 0.06                                           # no single position above 6% of the book

SUBTYPES = {"Loan": (["Term Loan", "Revolver", "Trade Finance"], [0.50, 0.35, 0.15]),
            "Bond": (["Corporate Bond", "Government Bond"], [0.80, 0.20]),
            "Derivative": (["Interest Rate Swap", "Equity Option/TRS", "Commodity Swap"], [0.45, 0.35, 0.20]),
            "Equity": (["Listed Equity"], [1.0])}
RATINGS = {"Loan": (["AA", "A", "BBB", "BB", "B"], [.05, .25, .40, .20, .10]),
           "Bond": (["AAA", "AA", "A", "BBB", "BB"], [.10, .15, .30, .35, .10]),
           "Derivative": (["AA", "A", "BBB"], [.20, .50, .30]),
           "Equity": (["NR"], [1.0])}
RATING_MULT = {"AAA": .4, "AA": .55, "A": .75, "BBB": 1.0, "BB": 1.5, "B": 2.1, "NR": 1.0}
GDP_BASE = {"Loan": .012, "Bond": .010, "Derivative": .004, "Equity": .025}
CYCLICAL = dict(zip(C.SECTORS, [1.1, 1.2, 1.2, 1.1, 0.7, 0.9]))
OIL_BASE = {"Energy": 0.60, "Consumer": -0.08, "Others": -0.06}      # other sectors ~0
OIL_TYPE = {"Loan": 0.35, "Bond": 0.25, "Equity": 1.0, "Derivative": 0.0}

NAME_STEMS = ["Nova", "Helix", "Quanta", "Vertex", "Cirrus", "Aperture", "Lumen", "Orbit", "Atlas", "Beacon",
              "Cobalt", "Dynamo", "Ember", "Falcon", "Granite", "Harbor", "Ionic", "Juniper", "Keystone", "Lattice",
              "Meridian", "Northwind", "Onyx", "Pinnacle", "Quartz", "Redwood", "Summit", "Tidal", "Umbra", "Vanguard"]
NAME_SUFFIX = {"Technology": ["Systems", "Cloud", "Semiconductor", "Software"],
               "Financials": ["Bank", "Capital", "Insurance", "Securities"],
               "Consumer": ["Retail", "Foods", "Brands", "Stores"],
               "Energy": ["Energy", "Petroleum", "Power", "Pipelines"],
               "Healthcare": ["Pharma", "Biosciences", "Health", "Medical"],
               "Others": ["Industries", "Logistics", "Utilities", "Aerospace"]}


def _counterparties(rng, per_sector=8):
    """Fictional counterparty names per sector, with a few used far more often than others."""
    pools = {}
    for sector in C.SECTORS:
        stems = rng.choice(NAME_STEMS, per_sector, replace=False)
        names = [f"{st} {rng.choice(NAME_SUFFIX[sector])}" for st in stems]
        w = 1 / (np.arange(per_sector) + 1) ** 0.9
        pools[sector] = (names, w / w.sum())
    return pools


def _sizes(rng, n, share, total):
    """Heavy-tailed sizes that add up to share x total, no position above the concentration limit."""
    v = rng.lognormal(0, 0.9, n)
    v = v / v.sum() * share * total
    cap = MAX_POSITION * total
    for _ in range(20):
        over = v > cap
        if not over.any():
            break
        excess = (v[over] - cap).sum()
        v[over] = cap
        room = ~over
        v[room] += excess * v[room] / v[room].sum()
    return v


def _sensitivities(rng, typ, sub, sector, rating):
    mult, cyc = RATING_MULT[rating], CYCLICAL[sector]
    gdp = GDP_BASE[typ] * mult * cyc * rng.uniform(0.8, 1.2)
    # interest rates
    if typ == "Loan":
        dur = {"Term Loan": rng.uniform(2, 4), "Revolver": rng.uniform(0.2, 0.8), "Trade Finance": rng.uniform(0.2, 0.6)}[sub]
    elif typ == "Bond":
        dur = rng.uniform(5, 9) if sub == "Government Bond" else rng.uniform(3, 8)
    elif typ == "Derivative":
        dur = rng.choice([-1, 1]) * rng.uniform(2, 7) if sub == "Interest Rate Swap" else 0.0   # pay-fixed swaps gain when rates rise
    else:
        dur = rng.uniform(2, 5) * (1.3 if sector == "Technology" else 1.0)
    # oil
    oil = OIL_BASE.get(sector, 0.01) * OIL_TYPE[typ] * rng.uniform(0.7, 1.3)
    if sub == "Commodity Swap":
        oil = rng.choice([1, -1], p=[0.7, 0.3]) * rng.uniform(0.8, 1.2)
    # VIX (market fear)
    if typ == "Equity":
        vix = -0.18 * rng.uniform(0.8, 1.3) * (1.15 if sector == "Technology" else 1.0)
    elif typ == "Loan":
        vix = -0.02
    elif typ == "Bond":
        vix = -0.03
    elif sub == "Equity Option/TRS":
        vix = rng.choice([1, -1]) * rng.uniform(0.12, 0.22)          # protective puts gain, total-return swaps lose
    else:
        vix = 0.0
    return round(dur, 2), round(oil, 3), round(vix, 3), round(gdp, 4)


def generate(n=None, seed=None, total=None):
    n, seed, total = n or C.PORTFOLIO_SIZE, seed if seed is not None else C.PORTFOLIO_SEED, total or C.PORTFOLIO_TOTAL_USD
    rng = np.random.default_rng(seed)
    pools = _counterparties(rng)

    counts = {t: int(round(COUNT_SHARE[t] * n)) for t in TYPES}
    counts["Loan"] += n - sum(counts.values())                    # make the counts add up exactly

    rows = []
    for typ in TYPES:
        sizes = _sizes(rng, counts[typ], VALUE_SHARE[typ], total)
        subs, sub_p = SUBTYPES[typ]
        for size in sizes:
            sub = rng.choice(subs, p=sub_p)
            if sub == "Commodity Swap":
                sector = "Energy"
            elif sub == "Government Bond":
                sector = "Others"
            else:
                sector = rng.choice(C.SECTORS, p=list(SECTOR_SHARE.values()))
            rating = "AA" if sub == "Government Bond" else rng.choice(RATINGS[typ][0], p=RATINGS[typ][1])
            names, w = pools[sector]
            cp = "Sovereign" if sub == "Government Bond" else rng.choice(names, p=w)
            dur, oil, vix, gdp = _sensitivities(rng, typ, sub, sector, rating)
            maturity = {"Loan": (1, 7), "Bond": (2, 10), "Derivative": (1, 10)}.get(typ)
            rows.append(dict(type=typ, subtype=sub, counterparty=cp, sector=sector, rating=rating,
                             market_value=round(float(size), 0),
                             maturity_years=round(rng.uniform(*maturity), 1) if maturity else np.nan,
                             rate_dur=dur, oil_beta=oil, vix_beta=vix, gdp_beta=gdp))
    df = pd.DataFrame(rows).sort_values("market_value", ascending=False).reset_index(drop=True)
    df.insert(0, "asset_id", [f"A{i + 1:03d}" for i in range(len(df))])
    df["weight"] = (df.market_value / df.market_value.sum()).round(4)
    return df


def save(df):
    C.ensure_dirs()
    df.to_csv(C.PORTFOLIO_CSV, index=False)


def load_portfolio():
    """Read the saved book, or generate (and save) it the first time."""
    if C.PORTFOLIO_CSV.exists():
        return pd.read_csv(C.PORTFOLIO_CSV)
    df = generate()
    save(df)
    return df


def summary(df):
    """Facts for the dashboard / README."""
    by = lambda col: {k: round(float(v), 4) for k, v in (df.groupby(col).market_value.sum() / df.market_value.sum()).items()}
    return {"positions": int(len(df)), "total_value": float(df.market_value.sum()), "by_type": by("type"), "by_sector": by("sector"),
            "top10_share": round(float(df.market_value.nlargest(10).sum() / df.market_value.sum()), 4),
            "counterparties": int(df.counterparty.nunique())}


if __name__ == "__main__":
    book = generate()
    save(book)
    print(summary(book))