"""Checks that everything is installed and in place, and tells you how to fix what is not.
Run from the project root:   python scripts/check_setup.py
"""
import sys
from importlib import metadata, util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (name you import, name you pip install)
PACKAGES = [("feedparser", "feedparser"), ("requests", "requests"), ("pandas", "pandas"), ("numpy", "numpy"),
            ("pyarrow", "pyarrow"), ("dotenv", "python-dotenv"), ("yfinance", "yfinance"),
            ("vaderSentiment", "vaderSentiment"), ("transformers", "transformers"), ("torch", "torch"),
            ("spacy", "spacy"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn")]
SRC_FILES = ["config", "ingest", "sentiment", "validate_sentiment", "entities", "events", "engine", "aggregate",
             "portfolio", "stress", "prices", "rebalancer", "api"]
problems = []


def line(ok, text, fix=None):
    print(("  OK       " if ok else "  MISSING  ") + text)
    if not ok and fix:
        problems.append(fix)


print(f"Python {sys.version.split()[0]} at {sys.executable}")
print("   (this must be the same Python your editor uses, see 'Select Interpreter' in the guide)\n")

print("Packages")
for mod, dist in PACKAGES:
    found = util.find_spec(mod) is not None
    try:
        ver = metadata.version(dist) if found else ""
    except metadata.PackageNotFoundError:
        ver = ""
    line(found, f"{dist} {ver}", f"pip install {dist}")

print("\nspaCy language model")
model_ok = False
if util.find_spec("spacy"):
    import spacy.util
    model_ok = spacy.util.is_package("en_core_web_sm")
line(model_ok, "en_core_web_sm", "python -m spacy download en_core_web_sm")

print("\nProject files")
for name in SRC_FILES:
    line((ROOT / "src" / f"{name}.py").exists(), f"src/{name}.py", f"save src/{name}.py")
line((ROOT / "src" / "__init__.py").exists(), "src/__init__.py (can be an empty file)", "create an empty file src/__init__.py")

print("\nSettings (.env)")
try:
    from src import config as C
    line(bool(C.NEWSAPI_KEY), "NEWSAPI_KEY is set", "add NEWSAPI_KEY=... to .env (NewsAPI is skipped without it)")
    line("your_id@college" not in C.SEC_USER_AGENT, "SEC_USER_AGENT has your name and email", "put your real name and email in SEC_USER_AGENT in .env")
    print(f"  info     mode={C.MODE}, auto refresh={'on' if C.AUTO_REFRESH else 'off'} every {C.REFRESH_MINUTES} min")
except Exception as ex:
    line(False, f"could not read src/config.py ({type(ex).__name__}: {ex})", "fix src/config.py first")

print("\nData")
line((ROOT / "data/raw/financial_news.csv").exists(), "data/raw/financial_news.csv", "see the Kaggle steps (python scripts/make_kaggle_samples.py)")
tw = (ROOT / "data/raw/stock_tweets.csv").exists()
print(("  OK       " if tw else "  optional ") + "data/raw/stock_tweets.csv" + ("" if tw else "   (the pipeline does not use it yet)"))
for rel in ["data/processed/signals.csv", "data/processed/prices_1d.csv", "data/processed/portfolio.csv"]:
    print(("  OK       " if (ROOT / rel).exists() else "  not yet  ") + rel + ("" if (ROOT / rel).exists() else "   (created by a later step)"))

print()
if problems:
    print("To fix:")
    for p in dict.fromkeys(problems):
        print("  " + p)
else:
    print("All good. Next: python -m src.validate_sentiment 300")