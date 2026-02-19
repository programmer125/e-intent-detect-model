# e-intent-detect-model

Conda single-environment setup for intent training and inference on Ubuntu 22.04 + CUDA 11.8.

## 1) Create environment

```bash
conda env create -f environment.yml
conda activate intent-llm
pip install -r requirements-cu118.txt
```

Alternative one-command setup:

```bash
bash scripts/setup_conda_env.sh
```

## 2) GPU validation

```bash
python scripts/verify_gpu.py
```

## 3) Smoke train (100 steps)

```bash
python scripts/smoke_train.py \
  --train_path data/intent_train_1k.train.jsonl \
  --val_path data/intent_train_1k.val.jsonl \
  --max_steps 100 \
  --model_source modelscope \
  --model_name damo/nlp_structbert_backbone_base_std
```

## 4) Inference benchmark (P50/P95/P99)

```bash
python scripts/benchmark_inference.py \
  --model_dir outputs/smoke_model \
  --input_path data/intent_train_1k.test.jsonl \
  --num_samples 1000 \
  --batch_size 1 \
  --max_length 64
```

## 5) ONNX export + consistency check

```bash
python scripts/export_onnx.py \
  --model_dir outputs/smoke_model \
  --onnx_path outputs/smoke_model/model.onnx \
  --max_length 64

python scripts/check_onnx_consistency.py \
  --model_dir outputs/smoke_model \
  --onnx_path outputs/smoke_model/model.onnx \
  --input_path data/intent_train_1k.test.jsonl \
  --num_samples 1000 \
  --max_length 64
```

## Notes

- Target defaults: `batch_size=1`, `max_length<=64`.
- Online serving should use a distilled small encoder model, not direct LLM decoding.
- ONNX Runtime GPU is preferred for low-latency inference.
- If your network cannot access HuggingFace, keep `--model_source modelscope`.
