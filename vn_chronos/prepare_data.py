"""
Phase 2 — Convert CSV.gz OHLCV files to Arrow format for Chronos training.

Each symbol's close price series becomes one row in the Arrow file.
Applies an 80/20 time-based split (no lookahead): train uses first 80% of each
series, val uses the full series (evaluation happens on the held-out tail).

Output:
  <TRAIN_DATA_DIR>/train.arrow
  <TRAIN_DATA_DIR>/val.arrow
  <TRAIN_DATA_DIR>/data_meta.json

Usage:
    python -m vn_chronos.prepare_data
    DATA_DIR=/data TRAIN_DATA_DIR=/train python -m vn_chronos.prepare_data
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa

DATA_DIR = os.environ.get("DATA_DIR", "/root/reports/vn_training_data")
TRAIN_DATA_DIR = os.environ.get("TRAIN_DATA_DIR", "/root/reports/vn_chronos_train")
MIN_BARS = int(os.environ.get("MIN_BARS", "120"))  # need enough for train+val


def load_series(path: str) -> tuple[datetime, list[float]] | None:
    try:
        df = pd.read_csv(path, compression="gzip")
        df.columns = [c.lower() for c in df.columns]
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
        if len(df) < MIN_BARS:
            return None
        start = df["time"].iloc[0].to_pydatetime().replace(tzinfo=None)
        values = df["close"].astype("float32").tolist()
        return start, values
    except Exception:
        return None


def write_arrow(records: list[tuple[datetime, list[float]]], path: str) -> int:
    schema = pa.schema([
        ("start", pa.timestamp("s")),
        ("target", pa.list_(pa.float32())),
    ])
    table = pa.table(
        {
            "start": pa.array([r[0] for r in records], type=pa.timestamp("s")),
            "target": pa.array([r[1] for r in records], type=pa.list_(pa.float32())),
        },
        schema=schema,
    )
    with pa.ipc.new_file(path, schema) as writer:
        writer.write_table(table)
    return sum(len(r[1]) for r in records)


def main():
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv.gz"))
    print(f"[prepare] {len(files)} symbol files → {TRAIN_DATA_DIR}")

    train_records: list[tuple[datetime, list[float]]] = []
    val_records: list[tuple[datetime, list[float]]] = []
    skipped = 0

    for i, fname in enumerate(files, 1):
        result = load_series(os.path.join(DATA_DIR, fname))
        if result is None:
            skipped += 1
            continue
        start, values = result
        split = max(MIN_BARS, int(len(values) * 0.8))
        train_records.append((start, values[:split]))
        val_records.append((start, values))          # full series for eval
        if i % 200 == 0:
            print(f"[prepare]   {i}/{len(files)} ...")

    print(f"[prepare] Writing Arrow files...")
    train_bars = write_arrow(train_records, os.path.join(TRAIN_DATA_DIR, "train.arrow"))
    val_bars = write_arrow(val_records, os.path.join(TRAIN_DATA_DIR, "val.arrow"))

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": len(train_records),
        "skipped": skipped,
        "train_bars": train_bars,
        "val_bars": val_bars,
        "train_file": os.path.join(TRAIN_DATA_DIR, "train.arrow"),
        "val_file": os.path.join(TRAIN_DATA_DIR, "val.arrow"),
    }
    with open(os.path.join(TRAIN_DATA_DIR, "data_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"""
[prepare] Done.
  symbols:    {len(train_records):,}  (skipped {skipped})
  train bars: {train_bars:,}
  val bars:   {val_bars:,}
  est 60d windows: ~{max(0, train_bars - 60 * len(train_records)):,}
""")


if __name__ == "__main__":
    main()
