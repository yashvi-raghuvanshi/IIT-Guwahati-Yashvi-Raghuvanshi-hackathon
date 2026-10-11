"""Shrink the two big Kaggle downloads into small sample files that are safe to commit.

Put the ORIGINAL downloads (any name below) in data/kaggle_full/ and run from the project root:
    python scripts/make_kaggle_samples.py
It writes:
    data/raw/stock_tweets.csv     (about 1,000 tweets about stocks on our watchlist)
    data/raw/financial_news.csv   (about 1,000 labelled sentences, same positive/neutral/negative mix)
The full files in data/kaggle_full/ are NOT committed (see .gitignore).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C                      # noqa: E402
from src.ingest import load_kaggle_news          # noqa: E402

FULL = C.DATA / "kaggle_full"
N_TWEETS, N_NEWS, SEED = 1000, 1000, 7


def _find(*names):
    for n in names:
        if (FULL / n).exists():
            return FULL / n
    return None


def _kb(path):
    return f"{path.stat().st_size / 1024:.0f} KB"


def tweets():
    src = _find("stock_tweets.csv")
    if not src:
        print(f"[skip] tweets: put stock_tweets.csv in {FULL}")
        return
    df = pd.read_csv(src)
    df = df.dropna(subset=["Tweet", "Date"]).drop_duplicates("Tweet")
    ours = {c["yahoo"] for c in C.WATCHLIST.values()}
    mine = df[df["Stock Name"].isin(ours)]
    print(f"tweets: {len(df):,} rows in the file, {len(mine):,} about our watchlist ({sorted(mine['Stock Name'].unique())})")
    if len(mine) == 0:                           # no overlap: keep a random slice of everything instead
        mine = df
    # take the same number from each stock, so no single stock dominates
    per = max(1, N_TWEETS // mine["Stock Name"].nunique())
    shuffled = mine.sample(frac=1, random_state=SEED)
    out = shuffled[shuffled.groupby("Stock Name").cumcount() < per]
    out = out.sort_values("Date")[["Date", "Tweet", "Stock Name"] + (["Company Name"] if "Company Name" in out else [])]
    out["Tweet"] = out["Tweet"].str.slice(0, 280)
    C.RAW.mkdir(parents=True, exist_ok=True)
    dest = C.RAW / "stock_tweets.csv"
    out.to_csv(dest, index=False)
    print(f"  wrote {dest.relative_to(C.ROOT)}: {len(out):,} rows, {_kb(dest)}")


def news():
    src = _find("all-data.csv", "financial_news.csv", "data.csv")
    if not src:
        print(f"[skip] news: put all-data.csv in {FULL}")
        return
    df = load_kaggle_news(src)                   # handles files with or without a header row
    df = df[df.label.isin(["positive", "neutral", "negative"])].drop_duplicates("text")
    print(f"news: {len(df):,} labelled sentences; mix {df.label.value_counts(normalize=True).round(2).to_dict()}")
    shuffled = df.sample(frac=1, random_state=SEED)                          # shuffled, so labels end up mixed
    quota = (df.label.value_counts(normalize=True) * min(N_NEWS, len(df))).round().astype(int)   # same label mix as the full file
    out = shuffled[shuffled.groupby("label").cumcount() < shuffled.label.map(quota)]
    C.RAW.mkdir(parents=True, exist_ok=True)
    dest = C.RAW / "financial_news.csv"
    out.to_csv(dest, index=False)                # saved as UTF-8; the loader reads UTF-8 or Latin-1
    print(f"  wrote {dest.relative_to(C.ROOT)}: {len(out):,} rows, {_kb(dest)}, mix {out.label.value_counts(normalize=True).round(2).to_dict()}")


if __name__ == "__main__":
    tweets()
    news()
    total = sum(f.stat().st_size for f in (C.RAW / "stock_tweets.csv", C.RAW / "financial_news.csv") if f.exists())
    print(f"\nTotal committed data from Kaggle: {total / 1024:.0f} KB" + ("  (over 1 MB: lower N_TWEETS / N_NEWS)" if total > 1_048_576 else "  (fine)"))