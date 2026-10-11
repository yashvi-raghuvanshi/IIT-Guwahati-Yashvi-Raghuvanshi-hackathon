"""Check the sentiment model against the labelled Kaggle financial news file.

Run:  python -m src.validate_sentiment            (1000 random sentences)
      python -m src.validate_sentiment 300        (faster, 300 sentences)
Writes docs/sentiment_validation.md for your slides and README.
"""
import sys

import pandas as pd

from src import config as C
from src import sentiment as S
from src.ingest import load_kaggle_news

LABELS = ["negative", "neutral", "positive"]


def to_label(score, t):
    return "positive" if score >= t else "negative" if score <= -t else "neutral"


def evaluate(true, scores, t):
    pred = pd.Series([to_label(s, t) for s in scores], index=true.index)
    cm = pd.crosstab(true, pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
    per_class, f1s = {}, []
    for l in LABELS:
        tp = cm.loc[l, l]
        prec = tp / max(cm[l].sum(), 1)
        rec = tp / max(cm.loc[l].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[l] = (prec, rec, f1)
        f1s.append(f1)
    return dict(accuracy=float((pred == true).mean()), macro_f1=sum(f1s) / 3, cm=cm, per_class=per_class)


def main(n=1000, seed=7):
    df = load_kaggle_news()
    df = df[df.label.isin(LABELS)]
    if df.empty:
        print("Put the Kaggle file at data/raw/financial_news.csv first.")
        return
    df = df.sample(min(n, len(df)), random_state=seed)
    true = df.label

    models = {}
    main_be = S.get_backend()
    models[main_be.name] = S.score_texts(df.text.tolist(), main_be)
    if main_be.name != "vader":                                  # naive baseline for comparison
        try:
            models["vader"] = S.score_texts(df.text.tolist(), S.make_backend("vader"))
        except Exception:
            print("[info] VADER not installed, skipping baseline")

    lines = [f"# Sentiment validation ({len(df)} labelled sentences)\n",
             f"Majority-class baseline accuracy: {true.value_counts(normalize=True).max():.1%}\n"]
    for name, scores in models.items():
        r = evaluate(true, scores, C.SENT_POS)
        print(f"\n{name}: accuracy {r['accuracy']:.1%}, macro-F1 {r['macro_f1']:.2f} (threshold ±{C.SENT_POS})")
        print(r["cm"].to_string())
        lines += [f"\n## {name}: accuracy {r['accuracy']:.1%}, macro-F1 {r['macro_f1']:.2f} (threshold ±{C.SENT_POS})\n",
                  "Rows = true label, columns = predicted\n", "```", r["cm"].to_string(), "```\n",
                  "| class | precision | recall | F1 |\n|---|---|---|---|"]
        lines += [f"| {l} | {p:.2f} | {rc:.2f} | {f:.2f} |" for l, (p, rc, f) in r["per_class"].items()]

    best = models[main_be.name]
    print("\nThreshold check (does ±0.15 fit this data?):")
    for t in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        print(f"  ±{t:.2f}: accuracy {evaluate(true, best, t)['accuracy']:.1%}")

    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "sentiment_validation.md").write_text("\n".join(lines))
    print("\nSaved docs/sentiment_validation.md")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)