import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--onnx_path", type=Path, required=True)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    torch_model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()

    requested_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    try:
        session = ort.InferenceSession(str(args.onnx_path), providers=requested_providers)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "warning": "Failed to initialize CUDAExecutionProvider, fallback to CPUExecutionProvider.",
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        session = ort.InferenceSession(str(args.onnx_path), providers=["CPUExecutionProvider"])

    texts = read_texts(args.input_path, args.num_samples)
    if not texts:
        raise ValueError("No input texts found.")

    same = 0
    total = 0

    for text in texts:
        enc = tokenizer(
            [text],
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            torch_logits = torch_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            ).logits
        torch_pred = int(torch.argmax(torch_logits, dim=-1).item())

        ort_inputs = {
            "input_ids": enc["input_ids"].cpu().numpy().astype(np.int64),
            "attention_mask": enc["attention_mask"].cpu().numpy().astype(np.int64),
        }
        ort_logits = session.run(["logits"], ort_inputs)[0]
        ort_pred = int(np.argmax(ort_logits, axis=-1)[0])

        same += int(torch_pred == ort_pred)
        total += 1

    report = {
        "samples": total,
        "match": same,
        "consistency": round(same / max(total, 1), 6),
        "providers": session.get_providers(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
