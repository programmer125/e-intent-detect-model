import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_texts(path: Path, num_samples: int) -> List[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if text:
                texts.append(text)
            if len(texts) >= num_samples:
                break
    return texts


def percentile_ms(values_ms, p):
    return float(np.percentile(np.array(values_ms, dtype=np.float64), p))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    texts = read_texts(args.input_path, args.num_samples)
    if len(texts) == 0:
        raise ValueError("No input texts found.")

    def run_batch(batch_texts):
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)

    for _ in range(args.warmup_steps):
        run_batch(texts[: args.batch_size])
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    total_samples = 0
    start_all = time.perf_counter()

    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        t0 = time.perf_counter()
        run_batch(batch_texts)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms / len(batch_texts))
        total_samples += len(batch_texts)

    total_s = time.perf_counter() - start_all
    metrics = {
        "device": str(device),
        "samples": total_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "p50_ms": round(percentile_ms(latencies_ms, 50), 3),
        "p95_ms": round(percentile_ms(latencies_ms, 95), 3),
        "p99_ms": round(percentile_ms(latencies_ms, 99), 3),
        "qps": round(total_samples / total_s, 3),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
