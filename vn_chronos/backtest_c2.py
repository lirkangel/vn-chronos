"""
Zero-shot Chronos-2 backtest on VN-equity held-out val set.

Compares against our VN-Chronos (53.6% at 5-bar) and random (50%).

Usage:
    python -m vn_chronos.backtest_c2
    MODEL_ID=amazon/chronos-2 VAL_DATA=... EVAL_SERIES=200 python -m vn_chronos.backtest_c2

Environment:
    MODEL_ID          HuggingFace model ID     (default: amazon/chronos-2)
    VAL_DATA          val.arrow path           (default: /root/reports/vn_chronos_train/val.arrow)
    EVAL_SERIES       series to sample         (default: 200)
    WINDOWS_PER       rolling windows/series   (default: 5)
    PREDICTION_LENGTH bars to forecast         (default: 5)
    CONTEXT_LENGTH    context bars             (default: 512)
    SEED                                       (default: 42)
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import torch
from chronos import Chronos2Pipeline

MODEL_ID = os.environ.get("MODEL_ID", "amazon/chronos-2")
VAL_DATA = os.environ.get("VAL_DATA", "/root/reports/vn_chronos_train/val.arrow")
EVAL_SERIES = int(os.environ.get("EVAL_SERIES", "200"))
WINDOWS_PER = int(os.environ.get("WINDOWS_PER", "5"))
PREDICTION_LENGTH = int(os.environ.get("PREDICTION_LENGTH", "5"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "512"))
SEED = int(os.environ.get("SEED", "42"))


def load_series(path: str) -> list[np.ndarray]:
    print(f"[c2-backtest] Loading: {path}")
    with pa.ipc.open_file(path) as f:
        table = f.read_all()
    series = []
    min_len = CONTEXT_LENGTH + PREDICTION_LENGTH
    for i in range(len(table)):
        values = table["target"][i].as_py()
        arr = np.array(values, dtype=np.float32)
        if len(arr) >= min_len and np.all(np.isfinite(arr)) and np.all(arr > 0):
            series.append(arr)
    print(f"[c2-backtest] {len(series)} usable series (>= {min_len} bars)")
    return series


def run_backtest(pipeline: Chronos2Pipeline, series: list[np.ndarray], rng: random.Random) -> dict:
    correct = 0
    naive_correct = 0
    total = 0
    upward_actual = 0
    errors = 0

    for idx, s in enumerate(series):
        if (idx + 1) % 20 == 0:
            rate = correct / total if total else 0
            print(f"  [{idx+1}/{len(series)}] accuracy so far: {rate:.1%} ({total} windows)")

        max_start = len(s) - CONTEXT_LENGTH - PREDICTION_LENGTH
        starts = [rng.randint(0, max_start) for _ in range(WINDOWS_PER)]

        for start in starts:
            ctx = s[start: start + CONTEXT_LENGTH]
            future = s[start + CONTEXT_LENGTH: start + CONTEXT_LENGTH + PREDICTION_LENGTH]

            last_price = float(ctx[-1])
            actual_end = float(future[-1])
            actual_up = actual_end > last_price
            upward_actual += int(actual_up)

            try:
                # shape: (n_series=1, n_variates=1, history_length)
                context_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # returns (list[quantile_tensors], list[mean_tensors])
                # quantile_tensors[0] shape: (n_variates=1, pred_length, n_quantiles=1)
                q_list, _ = pipeline.predict_quantiles(
                    context_tensor,
                    prediction_length=PREDICTION_LENGTH,
                    quantile_levels=[0.5],
                )
                median_end = float(q_list[0][0, -1, 0])
                predicted_up = median_end > last_price
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    print(f"  [warn] inference error: {exc}")
                continue

            correct += int(predicted_up == actual_up)
            naive_correct += int(actual_up)
            total += 1

    return {
        "total_windows": total,
        "errors": errors,
        "c2_accuracy": correct / total if total else 0,
        "naive_up_accuracy": naive_correct / total if total else 0,
        "actual_upward_rate": upward_actual / total if total else 0,
        "random_baseline": 0.5,
    }


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"[c2-backtest] Loading {MODEL_ID}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print(f"[c2-backtest] Model loaded. Params: {sum(p.numel() for p in pipeline.model.parameters()) / 1e6:.1f}M")

    series = load_series(VAL_DATA)
    rng = random.Random(SEED)
    sample = rng.sample(series, min(EVAL_SERIES, len(series)))

    print(f"[c2-backtest] {len(sample)} series × {WINDOWS_PER} windows = "
          f"{len(sample) * WINDOWS_PER} windows | pred_length={PREDICTION_LENGTH}\n")

    t0 = time.time()
    results = run_backtest(pipeline, sample, rng)
    elapsed = time.time() - t0

    edge = results["c2_accuracy"] - 0.5
    vs_naive = results["c2_accuracy"] - results["naive_up_accuracy"]

    print("\n" + "=" * 55)
    print("  CHRONOS-2 ZERO-SHOT BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Model             : {MODEL_ID}")
    print(f"  Windows           : {results['total_windows']}  (errors: {results['errors']})")
    print(f"  Actual upward rate: {results['actual_upward_rate']:.1%}")
    print()
    print(f"  Chronos-2 (zero-shot) : {results['c2_accuracy']:.1%}  ← model")
    print(f"  Naive 'always up'     : {results['naive_up_accuracy']:.1%}")
    print(f"  Random baseline       : 50.0%")
    print(f"  VN-Chronos trained    : 53.6%  (our 5-bar model, for reference)")
    print()
    print(f"  Edge vs random        : {edge:+.1%}")
    print(f"  Edge vs naive-up      : {vs_naive:+.1%}")
    print(f"\n  Elapsed: {elapsed:.0f}s")
    print("=" * 55)

    out_dir = os.environ.get("MODEL_DIR", "/root/reports/vn_chronos_model")
    out = {
        **results,
        "model_id": MODEL_ID,
        "eval_series": len(sample),
        "windows_per": WINDOWS_PER,
        "prediction_length": PREDICTION_LENGTH,
        "context_length": CONTEXT_LENGTH,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = os.path.join(out_dir, "backtest_c2_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[c2-backtest] Saved → {out_path}")


if __name__ == "__main__":
    main()
