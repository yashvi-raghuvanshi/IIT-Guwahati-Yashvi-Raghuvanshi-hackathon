"""Engine: runs the whole pipeline and writes the signal file everyone else reads.

    raw text -> company + sector -> event type -> merge repeated stories
             -> sentiment -> impact -> data/processed/signals.parquet (+ .csv)

Run:  python -m src.engine replay      (use what earlier live runs recorded)
      python -m src.engine live        (fetch fresh data first, then process)
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone

import pandas as pd

from src import config as C
from src import entities, events, ingest, sentiment

_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "as", "at", "by", "is", "are", "its", "it",
         "with", "from", "after", "says", "said", "amid", "over", "new", "stock", "shares", "inc", "corp"}


# ------------------------------------------------------------------ merge repeated stories
def _tokens(text):
    t = re.sub(r"\s+[-–—|]\s+[^-–—|]{2,40}$", "", text.lower())      # drop the " - Reuters" tail
    return {w for w in re.findall(r"[a-z0-9$%]+", t) if len(w) > 1 and w not in _STOP}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def merge_duplicates(df):
    """Keep the earliest copy of a story. Same company + event, similar words, within a few hours."""
    df = df.sort_values("ts").reset_index(drop=True)
    toks = [_tokens(t) for t in df.text]
    window = pd.Timedelta(hours=C.DEDUP_WINDOW_H)
    kept, kept_by_group = [], {}
    for i, row in enumerate(df.itertuples()):
        group = kept_by_group.setdefault((row.entity, row.event), [])
        is_dup = False
        for ts_k, tok_k in reversed(group):               # newest first, stop once outside the window
            if row.ts - ts_k > window:
                break
            if _jaccard(toks[i], tok_k) >= C.DEDUP_SIM:
                is_dup = True
                break
        if not is_dup:
            group.append((row.ts, toks[i]))
            kept.append(i)
    return df.iloc[kept].reset_index(drop=True), len(df) - len(kept)


# ------------------------------------------------------------------ the pipeline
def build_signals(raw):
    """raw = table from ingest. Returns (signals with the 11 contract columns, {'merged': n, 'dropped': n})."""
    ent = entities.link_df(raw)
    ev = events.classify_df(raw)
    df = pd.concat([raw[["id", "ts", "source", "text", "ticker_hint"]], ent, ev[["event"]]], axis=1)

    # a Google News search for "amazon stock" also returns deals and TV guides: it names no company and has no event
    noise = df.ticker_hint.isin(C.WATCHLIST) & (df.entity == C.MARKET_ENTITY) & (df.event == C.OTHER_EVENT)
    dropped = int(noise.sum())
    df = df[~noise]

    df, merged = merge_duplicates(df)                    # before sentiment: the slow step sees fewer texts

    df["sentiment"] = sentiment.score_df(df).astype(float).round(4)
    df["sentiment_label"] = df.sentiment.map(C.label_sentiment)
    df["impact"] = events.impact_df(df)

    out = df.rename(columns={"id": "signal_id"})[C.SIGNAL_COLUMNS]
    return out.sort_values("ts").reset_index(drop=True), {"merged": merged, "dropped": dropped}


# ------------------------------------------------------------------ save / load
def save_signals(out, meta):
    C.ensure_dirs()
    out.to_csv(C.SIGNALS_CSV, index=False)
    try:
        out.to_parquet(C.SIGNALS_PARQUET, index=False)
    except ImportError:
        print("[warn] pyarrow not installed: wrote CSV only (pip install pyarrow for the faster Parquet file)")
    C.ENGINE_META.write_text(json.dumps(meta, indent=2))


def load_signals():
    """What Module A, Module B and the dashboard call to read the engine output."""
    if C.SIGNALS_PARQUET.exists():
        try:
            return pd.read_parquet(C.SIGNALS_PARQUET)
        except ImportError:
            pass
    df = pd.read_csv(C.SIGNALS_CSV, dtype={"signal_id": str})
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    return df


# ------------------------------------------------------------------ run
def run(mode=None, days=None):
    mode, days = mode or C.MODE, days or C.REPLAY_DAYS
    t0 = time.time()
    if mode == "live":
        ingest.fetch_live()                              # fetches and caches; replay below reads the cache
    raw = ingest.load_replay(days=days)
    if raw.empty:
        print("No recorded data yet. Run:  python -m src.engine live")
        return None

    out, stats = build_signals(raw)
    meta = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), mode=mode,
                window_days=days, signals=len(out), duplicates_merged=stats["merged"], irrelevant_dropped=stats["dropped"],
                ts_min=out.ts.min().isoformat(), ts_max=out.ts.max().isoformat(),
                sentiment_model=sentiment.get_backend().name,
                high_impact=int((out.impact >= C.STRESS_TRIGGER_IMPACT).sum()),
                by_source=out.source.value_counts().to_dict(), by_event=out.event.value_counts().to_dict())
    save_signals(out, meta)

    print(f"\n{len(out)} signals ({stats['merged']} repeated stories merged, {stats['dropped']} irrelevant search results dropped) from {out.ts.min():%b %d} to {out.ts.max():%b %d}"
          f" in {time.time() - t0:.0f}s, sentiment model: {meta['sentiment_model']}")
    print("By source:   ", meta["by_source"])
    print("By event:    ", meta["by_event"])
    print("Sentiment:   ", out.sentiment_label.value_counts().to_dict())
    print(f"High impact (>= {C.STRESS_TRIGGER_IMPACT:g}): {meta['high_impact']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["live", "replay"], default=None)
    ap.add_argument("--days", type=int, default=None, help="replay window in days (default from config)")
    args = ap.parse_args()
    run(args.mode, args.days)