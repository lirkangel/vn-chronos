"""
Phase 1 — Bulk historical OHLCV fetcher for Vietnamese stock training dataset.

Providers (tried in order):
  1. DNSE  — if DNSE_API_KEY + DNSE_API_SECRET are set (HMAC-SHA256 auth)
  2. vnstock VCI — public, rate-limited at ~20 req/min free tier

Each symbol is saved as <DATA_DIR>/<SYMBOL>.csv.gz.
Progress is checkpointed after every symbol — safe to interrupt and resume.

Usage:
    python -m vn_chronos.fetch_data                        # all HOSE + HNX
    python -m vn_chronos.fetch_data --exchange HOSE        # HOSE only
    python -m vn_chronos.fetch_data --symbols VIC,HPG,SHB  # specific symbols
    python -m vn_chronos.fetch_data --resume               # skip already done
    python -m vn_chronos.fetch_data --start 2010-01-01     # shorter history

Environment:
    DATA_DIR          — output directory (default: ./data)
    DNSE_API_KEY      — DNSE HMAC key ID (preferred provider when set)
    DNSE_API_SECRET   — DNSE HMAC secret
    DNSE_BASE_URL     — default: https://openapi.dnse.com.vn
    VNSTOCK_SOURCE    — vnstock source (default: VCI)
    FETCH_DELAY_SEC   — base delay between symbols (default: 1.0)
    MIN_BARS          — minimum rows to keep a symbol (default: 60)
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import warnings
from datetime import datetime, timezone
from uuid import uuid4
from urllib import parse

import pandas as pd
import requests

DATA_DIR = os.environ.get("DATA_DIR", "./data")
FETCH_DELAY_SEC = float(os.environ.get("FETCH_DELAY_SEC", "1.0"))
MIN_BARS = int(os.environ.get("MIN_BARS", "60"))
VNSTOCK_SOURCE = os.environ.get("VNSTOCK_SOURCE", "VCI")
DNSE_BASE_URL = os.environ.get("DNSE_BASE_URL", "https://openapi.dnse.com.vn").rstrip("/")
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
        print(f"[fetch] Listing() failed ({ex}), trying stock object fallback...")
        stock = vnstock.Vnstock().stock(symbol="VIC", source=VNSTOCK_SOURCE)
        df = stock.listing.symbols_by_exchange(exchange=exchange)

    if df is None or df.empty:
        raise RuntimeError(f"No symbols returned for exchange {exchange}")

    for col in ("ticker", "symbol", "Ticker", "Symbol"):
        if col in df.columns:
            return df[col].dropna().str.strip().str.upper().tolist()

    raise RuntimeError(f"No symbol column in listing df. Columns: {list(df.columns)}")


def get_symbols(exchanges: list[str], explicit: list[str] | None = None) -> list[str]:
    if explicit:
        return [s.strip().upper() for s in explicit]

    seen: set[str] = set()
    result: list[str] = []
    for ex in exchanges:
        for sym in _get_symbols_from_listing(ex):
            if sym not in seen:
                result.append(sym)
                seen.add(sym)
    print(f"[fetch] Total unique symbols: {len(result)}")
    return result


# ---------------------------------------------------------------------------
# DNSE provider
# ---------------------------------------------------------------------------

def _dnse_headers(api_key: str, api_secret: str, path: str) -> dict:
    date_val = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    nonce = uuid4().hex
    sig_str = f"(request-target): get {path}\ndate: {date_val}\nnonce: {nonce}"
    digest = hmac.new(api_secret.encode(), sig_str.encode(), hashlib.sha256).digest()
    signature = parse.quote(base64.b64encode(digest).decode(), safe="")
    return {
        "Date": date_val,
        "X-Signature": (
            f'Signature keyId="{api_key}",algorithm="hmac-sha256",'
            f'headers="(request-target) date",signature="{signature}",nonce="{nonce}"'
        ),
        "x-api-key": api_key,
    }


def _coerce_time(value) -> pd.Timestamp:
    if isinstance(value, (int, float)) or (isinstance(value, str) and str(value).isdigit()):
        raw = int(value)
        unit = "ms" if raw > 10_000_000_000 else "s"
        return pd.to_datetime(raw, unit=unit)
    return pd.to_datetime(value)


def _parse_dnse_payload(payload, symbol: str) -> list[dict]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        # TradingView-style {t:[], o:[], h:[], l:[], c:[], v:[]}
        if all(k in payload for k in ("t", "o", "h", "l", "c")):
            times = payload["t"]
            vols = payload.get("v") or [0] * len(times)
            return [{"time": times[i], "open": payload["o"][i], "high": payload["h"][i],
                     "low": payload["l"][i], "close": payload["c"][i],
                     "volume": vols[i] if i < len(vols) else 0}
                    for i in range(len(times))]
        for key in ("data", "items", "candles", "ohlc"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    if isinstance(payload, list):
        return payload
    return []


def fetch_ohlcv_dnse(symbol: str, start: str, end: str) -> pd.DataFrame:
    api_key = os.environ.get("DNSE_API_KEY", "")
    api_secret = os.environ.get("DNSE_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("DNSE_API_KEY/DNSE_API_SECRET not set")

    path = "/price/ohlc"
    qs = parse.urlencode({
        "type": "STOCK",
        "symbol": symbol.upper(),
        "resolution": "1D",
        "from": int(pd.Timestamp(start).timestamp()),
        "to": int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp()),
    })
    url = f"{DNSE_BASE_URL}{path}?{qs}"
    # Sign only the path without query string — DNSE spec
    headers = _dnse_headers(api_key, api_secret, path)
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("DNSE rate limit (429)")
    if resp.status_code >= 400:
        raise RuntimeError(f"DNSE HTTP {resp.status_code}: {resp.text[:200]}")

    rows = _parse_dnse_payload(resp.json(), symbol)
    if not rows:
        raise ValueError(f"DNSE returned no rows for {symbol}")

    normalized = []
    for row in rows:
        if isinstance(row, dict):
            t = row.get("time") or row.get("timestamp") or row.get("t")
            o = row.get("open") or row.get("o")
            h = row.get("high") or row.get("h")
            l = row.get("low") or row.get("l")
            c = row.get("close") or row.get("c")
            v = row.get("volume") or row.get("v") or 0
        elif isinstance(row, (list, tuple)) and len(row) >= 5:
            t, o, h, l, c = row[:5]
            v = row[5] if len(row) > 5 else 0
        else:
            continue
        if None in (t, o, h, l, c):
            continue
        normalized.append({"time": _coerce_time(t), "open": float(o), "high": float(h),
                            "low": float(l), "close": float(c), "volume": float(v or 0)})

    if not normalized:
        raise ValueError(f"DNSE rows had no parseable data for {symbol}")

    df = pd.DataFrame(normalized)
    df = df.sort_values("time").reset_index(drop=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]


# ---------------------------------------------------------------------------
# vnstock provider
# ---------------------------------------------------------------------------

def fetch_ohlcv_vnstock(symbol: str, start: str, end: str) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    import vnstock

    quote = vnstock.Quote(symbol=symbol, source=VNSTOCK_SOURCE)
    df = quote.history(start=start, end=end, interval="1D")
    if df is None or df.empty:
        raise ValueError(f"vnstock returned empty for {symbol}")

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    time_col = next((c for c in df.columns if c in ("time", "date", "tradingdate")), None)
    if time_col and time_col != "time":
        df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"vnstock missing columns {missing} for {symbol}")
    return df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unified fetch with retry + backoff
# ---------------------------------------------------------------------------

_dnse_available = bool(os.environ.get("DNSE_API_KEY") and os.environ.get("DNSE_API_SECRET"))
_use_vnstock = True  # always enabled as fallback; SystemExit from vnstock is caught below


DNSE_RATE_LIMIT_WAIT = int(os.environ.get("DNSE_RATE_LIMIT_WAIT", "65"))
DNSE_RATE_LIMIT_MAX_WAITS = int(os.environ.get("DNSE_RATE_LIMIT_MAX_WAITS", "10"))


def fetch_ohlcv(symbol: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    errors = []

    if _dnse_available:
        rate_limit_waits = 0
        while True:
            try:
                return fetch_ohlcv_dnse(symbol, start, end)
            except RuntimeError as ex:
                if "429" in str(ex):
                    if rate_limit_waits >= DNSE_RATE_LIMIT_MAX_WAITS:
                        errors.append(f"dnse: rate limit persisted after {rate_limit_waits} waits")
                        break
                    rate_limit_waits += 1
                    print(f"  [dnse] rate limited — waiting {DNSE_RATE_LIMIT_WAIT}s to reset window (attempt {rate_limit_waits}/{DNSE_RATE_LIMIT_MAX_WAITS})")
                    time.sleep(DNSE_RATE_LIMIT_WAIT)
                else:
                    errors.append(f"dnse: {ex}")
                    break
            except Exception as ex:
                errors.append(f"dnse: {ex}")
                break

    if _use_vnstock:
        for attempt in range(1, max_retries + 1):
            try:
                return fetch_ohlcv_vnstock(symbol, start, end)
            except SystemExit:
                # vnstock calls sys.exit() when rate limited — catch it so our process survives
                errors.append(f"vnstock({attempt}): rate limited (sys.exit caught)")
                time.sleep(FETCH_DELAY_SEC * 3)
            except Exception as ex:
                errors.append(f"vnstock({attempt}): {ex}")
                if attempt < max_retries:
                    time.sleep(FETCH_DELAY_SEC * 2)

    raise RuntimeError(f"all providers failed: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Progress + storage
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
    with open(os.path.join(data_dir, PROGRESS_FILE), "w") as f:
        json.dump(progress, f, indent=2)


def symbol_path(symbol: str, data_dir: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in symbol.upper())
    return os.path.join(data_dir, f"{safe}.csv.gz")


def save_symbol(symbol: str, df: pd.DataFrame, data_dir: str) -> None:
    df.to_csv(symbol_path(symbol, data_dir), index=False, compression="gzip")


# ---------------------------------------------------------------------------
# Main loop
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

    provider_label = "DNSE→vnstock" if _dnse_available else "vnstock"
    print(f"\n[fetch] {len(symbols)} symbols | {start} → {end} | provider={provider_label} | dir={data_dir}\n")

    total = len(symbols)
    done = skipped = errors = total_bars = 0

    for i, symbol in enumerate(symbols, 1):
        status = progress.get(symbol, "")

        if status == "done":
            path = symbol_path(symbol, data_dir)
            if os.path.exists(path):
                try:
                    bars = len(pd.read_csv(path, compression="gzip"))
                    total_bars += bars
                    done += 1
                    print(f"[{i:>4}/{total}] {symbol:<8} skip (done, {bars} bars)")
                    continue
                except Exception:
                    pass
        elif status == "skipped":
            skipped += 1
            print(f"[{i:>4}/{total}] {symbol:<8} skip (< {MIN_BARS} bars)")
            continue

        try:
            df = fetch_ohlcv(symbol, start=start, end=end)
            bars = len(df)
            if bars < MIN_BARS:
                progress[symbol] = "skipped"
                skipped += 1
                print(f"[{i:>4}/{total}] {symbol:<8} skipped ({bars} bars < {MIN_BARS})")
            else:
                save_symbol(symbol, df, data_dir)
                progress[symbol] = "done"
                done += 1
                total_bars += bars
                date_range = f"{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}"
                print(f"[{i:>4}/{total}] {symbol:<8} OK  {bars:>5} bars  {date_range}")
        except Exception as ex:
            short = str(ex)[:120]
            progress[symbol] = f"error: {short}"
            errors += 1
            print(f"[{i:>4}/{total}] {symbol:<8} ERR {short}")

        save_progress(data_dir, progress)
        time.sleep(FETCH_DELAY_SEC)

    print(f"""
[fetch] ── Summary ──────────────────────────────────────
  Symbols attempted : {total}
  Saved             : {done}
  Skipped (< bars)  : {skipped}
  Errors            : {errors}
  Total bars        : {total_bars:,}
  Est. training windows (~60d): ~{max(0, total_bars - 60 * done):,}
  Output            : {os.path.abspath(data_dir)}
""")

    summary = {
        "fetched_at": datetime.utcnow().isoformat(),
        "symbols_total": total,
        "symbols_saved": done,
        "symbols_skipped": skipped,
        "symbols_error": errors,
        "total_bars": total_bars,
        "data_dir": os.path.abspath(data_dir),
    }
    with open(os.path.join(data_dir, "dataset_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fetch historical OHLCV for VN stock training dataset.")
    ap.add_argument("--exchange", nargs="+", default=["HOSE", "HNX"],
                    help="Exchange(s): HOSE HNX UPCOM (default: HOSE HNX)")
    ap.add_argument("--symbols", default=None,
                    help="Comma-separated symbol list (overrides --exchange)")
    ap.add_argument("--start", default=START_DATE,
                    help=f"Start date (default: {START_DATE})")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="Output directory (default: ./data or $DATA_DIR)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-fetch all symbols ignoring existing progress")
    args = ap.parse_args()

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
