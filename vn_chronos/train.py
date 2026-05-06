"""
Phase 2 — Train Chronos-T5-small from scratch on VN equity daily close prices.

Architecture : T5ForConditionalGeneration (46M params)
Tokenizer    : MeanScaleUniformBins (4096 bins)
Data         : 1,628 VN equity symbols, daily, back to 2000
Training time: ~48 hrs on 2-core CPU for 50K steps

Resume: re-run the same command — automatically picks up the latest checkpoint.

Usage:
    python -m vn_chronos.train
    MAX_STEPS=50000 BATCH_SIZE=32 python -m vn_chronos.train

Environment:
    TRAIN_DATA_DIR  — Arrow files dir  (default: /root/reports/vn_chronos_train)
    MODEL_OUTPUT_DIR— checkpoint dir   (default: /root/reports/vn_chronos_model)
    CONTEXT_LENGTH  — context bars     (default: 512)
    PREDICTION_LENGTH— forecast bars   (default: 5)
    MAX_STEPS       — total grad steps (default: 50000)
    BATCH_SIZE      — samples/step     (default: 32)
    LEARNING_RATE                      (default: 0.001)
    SAVE_STEPS      — checkpoint freq  (default: 5000)
    LOG_STEPS       — log freq         (default: 200)
    MIN_PAST        — min context bars (default: 60)
    SEED                               (default: 42)
"""

import glob
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Iterator

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import T5Config, T5ForConditionalGeneration
from chronos import ChronosConfig

TRAIN_DATA_DIR = os.environ.get("TRAIN_DATA_DIR", "/root/reports/vn_chronos_train")
MODEL_OUTPUT_DIR = os.environ.get("MODEL_OUTPUT_DIR", "/root/reports/vn_chronos_model")
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "512"))
PREDICTION_LENGTH = int(os.environ.get("PREDICTION_LENGTH", "5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "50000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "5000"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "200"))
MIN_PAST = int(os.environ.get("MIN_PAST", "60"))
SEED = int(os.environ.get("SEED", "42"))

CHRONOS_CONFIG = ChronosConfig(
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
    num_samples=20,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
)

T5_CONFIG = T5Config(
    d_model=512,
    d_ff=2048,
    d_kv=64,
    num_heads=8,
    num_layers=6,
    num_decoder_layers=6,
    vocab_size=4096,
    pad_token_id=0,
    eos_token_id=1,
    decoder_start_token_id=0,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    is_encoder_decoder=True,
    initializer_factor=0.05,
    feed_forward_proj="relu",
    layer_norm_epsilon=1e-6,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_arrow_series(path: str) -> list[np.ndarray]:
    with pa.ipc.open_file(path) as f:
        table = f.read_all()
    series = []
    for i in range(len(table)):
        values = table["target"][i].as_py()
        arr = np.array(values, dtype=np.float32)
        min_len = MIN_PAST + PREDICTION_LENGTH
        if len(arr) >= min_len and np.all(np.isfinite(arr)) and np.all(arr > 0):
            series.append(arr)
    return series


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChronosWindowDataset(IterableDataset):
    """Infinite stream of random (context, future) windows sampled from series."""

    def __init__(self, series: list[np.ndarray], seed: int = SEED):
        self.series = series
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)

        while True:
            s = rng.choice(self.series)
            total = CONTEXT_LENGTH + PREDICTION_LENGTH
            if len(s) < total:
                ctx_len = rng.randint(MIN_PAST, len(s) - PREDICTION_LENGTH)
            else:
                max_start = len(s) - total
                start = rng.randint(0, max_start)
                s = s[start:start + total]
                ctx_len = rng.randint(MIN_PAST, CONTEXT_LENGTH)

            context = torch.tensor(s[:ctx_len], dtype=torch.float32)
            future = torch.tensor(s[ctx_len:ctx_len + PREDICTION_LENGTH], dtype=torch.float32)

            if len(future) < PREDICTION_LENGTH:
                continue

            yield {"context": context, "future": future}


# ---------------------------------------------------------------------------
# Collator — pads contexts, tokenizes batch
# ---------------------------------------------------------------------------

class ChronosCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        contexts = [b["context"] for b in batch]
        futures = torch.stack([b["future"] for b in batch])  # (B, pred_len)

        max_ctx = max(c.shape[0] for c in contexts)
        padded = torch.zeros(len(batch), max_ctx)
        pad_mask = torch.zeros(len(batch), max_ctx, dtype=torch.bool)
        for i, c in enumerate(contexts):
            l = c.shape[0]
            padded[i, max_ctx - l:] = c      # left-pad, recent values at right
            pad_mask[i, max_ctx - l:] = True

        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(padded)
        # Extend pad_mask to match tokenizer output length (EOS token makes it +1)
        if attention_mask.shape[1] > pad_mask.shape[1]:
            extra = attention_mask.shape[1] - pad_mask.shape[1]
            pad_mask = torch.cat(
                [pad_mask, torch.ones(len(batch), extra, dtype=torch.bool)], dim=1
            )
        attention_mask = attention_mask & pad_mask
        labels, _ = self.tokenizer.label_input_transform(futures, scale)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def latest_checkpoint(output_dir: str) -> tuple[str | None, int]:
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None, 0
    steps = []
    for cp in checkpoints:
        try:
            steps.append((int(cp.rsplit("-", 1)[-1]), cp))
        except ValueError:
            pass
    if not steps:
        return None, 0
    steps.sort()
    return steps[-1][1], steps[-1][0]


def save_checkpoint(model, tokenizer, optimizer, step: int, loss: float, output_dir: str):
    cp_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(cp_dir, exist_ok=True)
    model.save_pretrained(cp_dir)
    torch.save(optimizer.state_dict(), os.path.join(cp_dir, "optimizer.pt"))
    with open(os.path.join(cp_dir, "train_state.json"), "w") as f:
        json.dump({"step": step, "loss": loss, "saved_at": datetime.now(timezone.utc).isoformat()}, f)
    # Save chronos config for inference
    with open(os.path.join(cp_dir, "chronos_config.json"), "w") as f:
        json.dump({
            "tokenizer_class": "MeanScaleUniformBins",
            "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0},
            "n_tokens": 4096, "n_special_tokens": 2,
            "pad_token_id": 0, "eos_token_id": 1, "use_eos_token": True,
            "model_type": "seq2seq", "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "num_samples": 20, "temperature": 1.0, "top_k": 50, "top_p": 1.0,
        }, f, indent=2)
    print(f"[train] checkpoint saved → {cp_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print(f"[train] Loading training data from {TRAIN_DATA_DIR}...")
    train_path = os.path.join(TRAIN_DATA_DIR, "train.arrow")
    series = load_arrow_series(train_path)
    print(f"[train] Loaded {len(series)} series | total bars: {sum(len(s) for s in series):,}")

    tokenizer = CHRONOS_CONFIG.create_tokenizer()

    # Model — load from checkpoint or init from scratch
    cp_path, start_step = latest_checkpoint(MODEL_OUTPUT_DIR)
    if cp_path:
        print(f"[train] Resuming from checkpoint: {cp_path} (step {start_step})")
        model = T5ForConditionalGeneration.from_pretrained(cp_path)
    else:
        print("[train] Initializing T5-small from scratch...")
        model = T5ForConditionalGeneration(T5_CONFIG)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model params: {total_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    if cp_path:
        opt_path = os.path.join(cp_path, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

    dataset = ChronosWindowDataset(series)
    collator = ChronosCollator(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        num_workers=int(os.environ.get("DATALOADER_WORKERS", "2")),
        prefetch_factor=2,
    )

    remaining = MAX_STEPS - start_step
    print(f"\n[train] Training: steps {start_step} → {MAX_STEPS} "
          f"| batch={BATCH_SIZE} | lr={LEARNING_RATE}\n")

    model.train()
    step = start_step
    loss_acc = 0.0
    t0 = time.time()

    for batch in loader:
        if step >= MAX_STEPS:
            break

        optimizer.zero_grad()
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_acc += out.loss.item()
        step += 1

        if step % LOG_STEPS == 0:
            avg_loss = loss_acc / LOG_STEPS
            elapsed = time.time() - t0
            rate = LOG_STEPS / elapsed
            eta_sec = (MAX_STEPS - step) / rate
            eta_h = eta_sec / 3600
            print(f"[train] step {step:>6}/{MAX_STEPS}  loss={avg_loss:.4f}  "
                  f"{rate:.1f} steps/sec  ETA {eta_h:.1f}h")
            loss_acc = 0.0
            t0 = time.time()

        if step % SAVE_STEPS == 0:
            save_checkpoint(model, tokenizer, optimizer, step,
                            out.loss.item(), MODEL_OUTPUT_DIR)

    # Final save
    save_checkpoint(model, tokenizer, optimizer, step, out.loss.item(), MODEL_OUTPUT_DIR)
    print(f"\n[train] Done. Final checkpoint at step {step}.")


if __name__ == "__main__":
    main()
