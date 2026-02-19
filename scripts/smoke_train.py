import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
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


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--val_path", type=Path, required=True)
    parser.add_argument("--test_path", type=Path, default=None)
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
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=float, default=8.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = to_single_label(read_jsonl(args.train_path))
    val_rows = to_single_label(read_jsonl(args.val_path))
    test_rows = to_single_label(read_jsonl(args.test_path)) if args.test_path else []
    if not train_rows or not val_rows:
        raise ValueError("No valid single-label rows found. Check input jsonl files.")

    labels = sorted({x[1] for x in train_rows + val_rows + test_rows})
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
    test_ds = build_dataset(test_rows, label2id) if test_rows else None

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    if test_ds is not None:
        test_ds = test_ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=True,
        save_safetensors=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))

    test_metrics = {}
    test_report = {}
    if test_ds is not None:
        pred_output = trainer.predict(test_ds, metric_key_prefix="test")
        test_metrics = pred_output.metrics
        pred_ids = np.argmax(pred_output.predictions, axis=1)
        label_ids = pred_output.label_ids
        test_report = classification_report(
            label_ids,
            pred_ids,
            labels=list(range(len(id2label))),
            target_names=[id2label[i] for i in range(len(id2label))],
            output_dict=True,
            zero_division=0,
        )
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    train_summary = {
        "model_name": args.model_name,
        "model_source": args.model_source,
        "model_resolved_path": model_path,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "num_labels": len(labels),
        "args": vars(args),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    write_json(args.output_dir / "train_summary.json", train_summary)
    if test_report:
        write_json(args.output_dir / "test_classification_report.json", test_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
