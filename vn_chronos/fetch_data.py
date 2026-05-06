"""
Phase 1 — Bulk historical OHLCV fetcher for Vietnamese stock training dataset.

Fetches all stocks from HOSE (and optionally HNX) going back to 2000-01-01,
storing each symbol as <DATA_DIR>/<SYMBOL>.csv.gz.

Usage:
    python -m vn_chronos.fetch_data                        # all HOSE from 2000
    python -m vn_chronos.fetch_data --exchange HNX         # HNX instead
    python -m vn_chronos.fetch_data --exchange HOSE HNX    # both exchanges
    python -m vn_chronos.fetch_data --symbols VIC,HPG,SHB  # specific symbols
    python -m vn_chronos.fetch_data --resume               # skip already done
    python -m vn_chronos.fetch_data --start 2010-01-01     # shorter history

Environment:
    DATA_DIR        — output directory (default: ./data)
    FETCH_DELAY_SEC — sleep between requests (default: 1.0)
    MIN_BARS        — minimum rows to keep a symbol (default: 60)
    VNSTOCK_SOURCE  — vnstock data source (default: VCI)
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "./data")
FETCH_DELAY_SEC = float(os.environ.get("FETCH_DELAY_SEC", "1.0"))
MIN_BARS = int(os.environ.get("MIN_BARS", "60"))
VNSTOCK_SOURCE = os.environ.get("VNSTOCK_SOURCE", "VCI")
PROGRESS_FILE = "fetch_progress.json"
START_DATE = "2000-01-01"


# ---------------------------------------------------------------------------
# Symbol listing
# ---------------------------------------------------------------------------

def _get_symbols_from_listing(exchange: str) -> list[str]:
    warnings.filterwarnings("ignore")
    import vnstock

    print(f"[fetch] Fetching {exchange} symbol list via vnstock...")
    try:
        listing = vnstock.Listing()
        df = listing.symbols_by_exchange(exchange=exchange)
    except Exception as ex:
        # Fallback: try via stock object
        print(f"[fetch] Listing() failed ({ex}), trying stock object fallback...")
        stock = vnstock.Vnstock().stock(symbol="VIC", source=VNSTOCK_SOURCE)
        df = stock.listing.symbols_by_exchange(exchange=exchange)

    if df is None or df.empty:
        raise RuntimeError(f"No symbols returned for exchange {exchange}")

    # vnstock returns 'ticker' or 'symbol' depending on version
    for col in ("ticker", "symbol", "Ticker", "Symbol"):
        if col in df.columns:
            symbols = df[col].dropna().str.strip().str.upper().tolist()
            print(f"[fetch] Found {len(symbols)} symbols on {exchange}")
            return symbols

    raise RuntimeError(f"Cannot find symbol column in listing df. Columns: {list(df.columns)}")


def get_symbols(exchanges: list[str], explicit: list[str] | None = None) -> list[str]:
    if explicit:
        return [s.strip().upper() for s in explicit]

    all_symbols: list[str] = []
    seen: set[str] = set()
    for ex in exchanges:
        for sym in _get_symbols_from_listing(ex):
            if sym not in seen:
                all_symbols.append(sym)
                seen.add(sym)
    return all_symbols


# ---------------------------------------------------------------------------
# OHLCV fetching
# ---------------------------------------------------------------------------

def _normalize_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    # time column: could be 'time', 'date', 'tradingdate'
    time_col = next((c for c in df.columns if c in ("time", "date", "tradingdate")), None)
    if time_col is None:
        raise ValueError(f"{symbol}: no time/date column found. columns={list(df.columns)}")
    if time_col != "time":
        df = df.rename(columns={time_col: "time"})

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def fetch_ohlcv(symbol: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    import vnstock

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            quote = vnstock.Quote(symbol=symbol, source=VNSTOCK_SOURCE)
            df = quote.history(start=start, end=end, interval="1D")
            if df is None or df.empty:
                raise ValueError("empty response")
            return _normalize_df(df, symbol)
        except Exception as ex:
            last_err = ex
            if attempt < max_retries:
                time.sleep(FETCH_DELAY_SEC * 2)
    raise RuntimeError(f"fetch failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress(data_dir: str) -> dict:
    path = os.path.join(data_dir, PROGRESS_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress(data_dir: str, progress: dict) -> None:
    path = os.path.join(data_dir, PROGRESS_FILE)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def symbol_path(symbol: str, data_dir: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in symbol.upper())
    return os.path.join(data_dir, f"{safe}.csv.gz")


def save_symbol(symbol: str, df: pd.DataFrame, data_dir: str) -> str:
    path = symbol_path(symbol, data_dir)
    df.to_csv(path, index=False, compression="gzip")
    return path


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def run_fetch(
    exchanges: list[str] = ("HOSE",),
    explicit_symbols: list[str] | None = None,
    start: str = START_DATE,
    resume: bool = True,
    data_dir: str = DATA_DIR,
) -> dict:
    os.makedirs(data_dir, exist_ok=True)
    end = datetime.today().strftime("%Y-%m-%d")

    symbols = get_symbols(list(exchanges), explicit_symbols)
    progress = load_progress(data_dir) if resume else {}

    total = len(symbols)
    done_count = skipped_count = error_count = 0
    total_bars = 0

    print(f"\n[fetch] Starting fetch: {total} symbols | {start} → {end} | data_dir={data_dir}\n")

    for i, symbol in enumerate(symbols, 1):
        status = progress.get(symbol, "")
        if status == "done":
            path = symbol_path(symbol, data_dir)
            if os.path.exists(path):
                try:
                    existing = pd.read_csv(path, compression="gzip")
                    bars = len(existing)
                    total_bars += bars
                    done_count += 1
                    print(f"[{i:>4}/{total}] {symbol:<8} skip (done, {bars} bars)")
                    continue
                except Exception:
                    pass  # re-fetch if file is corrupted
        elif status == "skipped":
            skipped_count += 1
            print(f"[{i:>4}/{total}] {symbol:<8} skip (< {MIN_BARS} bars)")
            continue

        try:
            df = fetch_ohlcv(symbol, start=start, end=end)
            bars = len(df)

            if bars < MIN_BARS:
                progress[symbol] = "skipped"
                skipped_count += 1
                print(f"[{i:>4}/{total}] {symbol:<8} skipped ({bars} bars < {MIN_BARS})")
            else:
                save_symbol(symbol, df, data_dir)
                progress[symbol] = "done"
                done_count += 1
                total_bars += bars
                date_range = f"{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}"
                print(f"[{i:>4}/{total}] {symbol:<8} OK  {bars:>5} bars  {date_range}")

        except Exception as ex:
            short_err = str(ex)[:120]
            progress[symbol] = f"error: {short_err}"
            error_count += 1
            print(f"[{i:>4}/{total}] {symbol:<8} ERR {short_err}")

        save_progress(data_dir, progress)
        time.sleep(FETCH_DELAY_SEC)

    # Summary
    print(f"""
[fetch] ── Summary ─────────────────────────────────────
  Symbols attempted : {total}
  Saved             : {done_count}
  Skipped (< bars)  : {skipped_count}
  Errors            : {error_count}
  Total bars saved  : {total_bars:,}
  Est. training windows (~60d lookback): ~{max(0, total_bars - 60 * done_count):,}
  Data dir          : {os.path.abspath(data_dir)}
""")

    summary = {
        "fetched_at": datetime.utcnow().isoformat(),
        "symbols_total": total,
        "symbols_saved": done_count,
        "symbols_skipped": skipped_count,
        "symbols_error": error_count,
        "total_bars": total_bars,
        "data_dir": os.path.abspath(data_dir),
    }
    with open(os.path.join(data_dir, "dataset_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Fetch historical OHLCV for VN stock training dataset.")
    ap.add_argument("--exchange", nargs="+", default=["HOSE"],
                    help="Exchange(s) to fetch: HOSE HNX UPCOM (default: HOSE)")
    ap.add_argument("--symbols", default=None,
                    help="Comma-separated list of symbols to fetch instead of full exchange")
    ap.add_argument("--start", default=START_DATE,
                    help=f"Start date (default: {START_DATE})")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="Output directory (default: ./data or $DATA_DIR)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-fetch all symbols, ignoring existing progress")
    return ap.parse_args()


def main():
    args = _parse_args()
    explicit = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    run_fetch(
        exchanges=args.exchange,
        explicit_symbols=explicit,
        start=args.start,
        resume=not args.no_resume,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
