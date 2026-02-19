import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def resolve_model_path(model_name: str, model_source: str, cache_dir: Path) -> str:
    if model_source == "huggingface":
        return model_name
    if model_source == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "ModelScope is not installed. Run: pip install modelscope==1.18.1"
            ) from exc
        local_dir = snapshot_download(model_id=model_name, cache_dir=str(cache_dir))
        return local_dir
    raise ValueError(f"Unsupported model_source: {model_source}")


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_single_label(rows: List[Dict]) -> List[Tuple[str, str]]:
    out = []
    for row in rows:
        text = row.get("text", "").strip()
        label_ids = row.get("label_ids", [])
        if not text or not label_ids:
            continue
        out.append((text, label_ids[0]))
    return out


def build_dataset(rows: List[Tuple[str, str]], label2id: Dict[str, int]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [x[0] for x in rows],
            "labels": [label2id[x[1]] for x in rows],
        }
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--val_path", type=Path, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default="damo/nlp_structbert_backbone_base_std",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="modelscope",
    )
    parser.add_argument("--model_cache_dir", type=Path, default=Path(".cache/models"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/smoke_model"))
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    train_rows = to_single_label(read_jsonl(args.train_path))
    val_rows = to_single_label(read_jsonl(args.val_path))
    if not train_rows or not val_rows:
        raise ValueError("No valid single-label rows found. Check input jsonl files.")

    labels = sorted({x[1] for x in train_rows + val_rows})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    model_path = resolve_model_path(
        model_name=args.model_name,
        model_source=args.model_source,
        cache_dir=args.model_cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    train_ds = build_dataset(train_rows, label2id)
    val_ds = build_dataset(val_rows, label2id)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
