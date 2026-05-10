"""
Backtest VN-Chronos trained checkpoint on held-out val set.

Metric: direction accuracy — does the median forecast end above the last
        known close? Compared to naive "always up" and 50% random baselines.

Usage:
    python -m vn_chronos.backtest
    MODEL_DIR=... VAL_DATA=... EVAL_SERIES=200 python -m vn_chronos.backtest

Environment:
    MODEL_DIR         checkpoint dir        (default: /root/reports/vn_chronos_model)
    VAL_DATA          val.arrow path        (default: /root/reports/vn_chronos_train/val.arrow)
    EVAL_SERIES       series to sample      (default: 200)
    WINDOWS_PER       rolling windows/series(default: 5)
    NUM_SAMPLES       forecast samples      (default: 20)
    CONTEXT_LENGTH                          (default: 512)
    PREDICTION_LENGTH                       (default: 5)
    SEED                                    (default: 42)
"""

import glob
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import torch
from transformers import T5ForConditionalGeneration
from chronos import ChronosConfig

MODEL_DIR = os.environ.get("MODEL_DIR", "/root/reports/vn_chronos_model")
VAL_DATA = os.environ.get("VAL_DATA", "/root/reports/vn_chronos_train/val.arrow")
EVAL_SERIES = int(os.environ.get("EVAL_SERIES", "200"))
WINDOWS_PER = int(os.environ.get("WINDOWS_PER", "5"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "20"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "512"))
PREDICTION_LENGTH = int(os.environ.get("PREDICTION_LENGTH", "5"))
SEED = int(os.environ.get("SEED", "42"))


def latest_checkpoint(model_dir: str) -> str:
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoints:
        sys.exit(f"No checkpoints found in {model_dir}")
    steps = []
    for cp in checkpoints:
        try:
            steps.append((int(cp.rsplit("-", 1)[-1]), cp))
        except ValueError:
            pass
    steps.sort()
    return steps[-1][1]


def load_series(path: str) -> list[np.ndarray]:
    print(f"[backtest] Loading: {path}")
    with pa.ipc.open_file(path) as f:
        table = f.read_all()
    series = []
    min_len = CONTEXT_LENGTH + PREDICTION_LENGTH
    for i in range(len(table)):
        values = table["target"][i].as_py()
        arr = np.array(values, dtype=np.float32)
        if len(arr) >= min_len and np.all(np.isfinite(arr)) and np.all(arr > 0):
            series.append(arr)
    print(f"[backtest] {len(series)} usable series (>= {min_len} bars)")
    return series


def make_tokenizer():
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        num_samples=NUM_SAMPLES,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    return config.create_tokenizer()


def forecast(model, tokenizer, context: np.ndarray) -> np.ndarray:
    """Returns (NUM_SAMPLES, PREDICTION_LENGTH) array of forecasted close prices."""
    ctx = torch.tensor(context[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0)

    T = ctx.shape[1]
    padded = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.float32)
    pad_mask = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.bool)
    padded[0, CONTEXT_LENGTH - T:] = ctx[0]
    pad_mask[0, CONTEXT_LENGTH - T:] = True

    input_ids, attention_mask, scale = tokenizer.context_input_transform(padded)

    if attention_mask.shape[1] > pad_mask.shape[1]:
        extra = attention_mask.shape[1] - pad_mask.shape[1]
        pad_mask = torch.cat([pad_mask, torch.ones(1, extra, dtype=torch.bool)], dim=1)
    attention_mask = attention_mask & pad_mask

    input_ids_rep = input_ids.repeat_interleave(NUM_SAMPLES, dim=0)
    attn_rep = attention_mask.repeat_interleave(NUM_SAMPLES, dim=0)

    with torch.no_grad():
        sample_ids = model.generate(
            input_ids=input_ids_rep,
            attention_mask=attn_rep,
            decoder_start_token_id=0,
            max_new_tokens=PREDICTION_LENGTH,
            do_sample=True,
            temperature=1.0,
            top_k=50,
        )

    if sample_ids.shape[1] > PREDICTION_LENGTH:
        sample_ids = sample_ids[:, -PREDICTION_LENGTH:]

    sample_ids_3d = sample_ids.unsqueeze(0)   # (1, S, pred_len)
    result = tokenizer.output_transform(sample_ids_3d, scale)  # (1, S, pred_len)
    return result[0].numpy()  # (S, pred_len)


def run_backtest(model, tokenizer, series: list[np.ndarray], rng: random.Random) -> dict:
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
                preds = forecast(model, tokenizer, ctx)   # (S, pred_len)
                median_end = float(np.median(preds[:, -1]))
                predicted_up = median_end > last_price
            except Exception as exc:
                errors += 1
                continue

            correct += int(predicted_up == actual_up)
            naive_correct += int(actual_up)
            total += 1

    return {
        "total_windows": total,
        "errors": errors,
        "vn_chronos_accuracy": correct / total if total else 0,
        "naive_up_accuracy": naive_correct / total if total else 0,
        "actual_upward_rate": upward_actual / total if total else 0,
        "random_baseline": 0.5,
    }


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    cp_path = latest_checkpoint(MODEL_DIR)
    print(f"[backtest] Checkpoint: {cp_path}")

    print("[backtest] Loading model…")
    model = T5ForConditionalGeneration.from_pretrained(cp_path)
    model.eval()
    print(f"[backtest] Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    tokenizer = make_tokenizer()
    series = load_series(VAL_DATA)
    rng = random.Random(SEED)
    sample = rng.sample(series, min(EVAL_SERIES, len(series)))

    print(f"[backtest] {len(sample)} series × {WINDOWS_PER} windows = "
          f"{len(sample) * WINDOWS_PER} windows | samples={NUM_SAMPLES}\n")

    t0 = time.time()
    results = run_backtest(model, tokenizer, sample, rng)
    elapsed = time.time() - t0

    edge = results["vn_chronos_accuracy"] - 0.5
    vs_naive = results["vn_chronos_accuracy"] - results["naive_up_accuracy"]

    print("\n" + "=" * 55)
    print("  VN-CHRONOS BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Windows           : {results['total_windows']}  (errors: {results['errors']})")
    print(f"  Actual upward rate: {results['actual_upward_rate']:.1%}")
    print()
    print(f"  VN-Chronos        : {results['vn_chronos_accuracy']:.1%}  ← model")
    print(f"  Naive 'always up' : {results['naive_up_accuracy']:.1%}")
    print(f"  Random baseline   : 50.0%")
    print()
    print(f"  Edge vs random    : {edge:+.1%}")
    print(f"  Edge vs naive-up  : {vs_naive:+.1%}")
    print(f"\n  Elapsed: {elapsed:.0f}s")
    print("=" * 55)

    out = {
        **results,
        "checkpoint": cp_path,
        "eval_series": len(sample),
        "windows_per": WINDOWS_PER,
        "num_samples": NUM_SAMPLES,
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = os.path.join(MODEL_DIR, "backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[backtest] Saved → {out_path}")


if __name__ == "__main__":
    main()
