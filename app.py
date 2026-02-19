import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "outputs" / "smoke_model")))
ONNX_PATH = Path(os.getenv("ONNX_PATH", str(MODEL_DIR / "model.onnx")))
INTENT_PATH = Path(os.getenv("INTENT_PATH", str(BASE_DIR / "intent" / "common.json")))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "64"))
OTHER_THRESHOLD = float(os.getenv("OTHER_THRESHOLD", "0.60"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))


class PredictRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)


class PredictResponse(BaseModel):
    intent_id: str
    level1: Optional[str] = None
    level2: Optional[str] = None
    score: float
    is_other: bool
    model_version: str


class ModelArtifacts:
    def __init__(self) -> None:
        self.model_dir: Path = MODEL_DIR
        self.onnx_path: Path = ONNX_PATH
        self.intent_path: Path = INTENT_PATH
        self.max_length: int = MAX_LENGTH
        self.other_threshold: float = OTHER_THRESHOLD
        self.model_version: str = os.getenv("MODEL_VERSION", self.model_dir.name)
        self.tokenizer = None
        self.session = None
        self.id2label: Dict[int, str] = {}
        self.intent_reverse: Dict[str, Dict[str, str]] = {}
        self.providers: List[str] = []

    def load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"MODEL_DIR not found: {self.model_dir}")
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX_PATH not found: {self.onnx_path}")
        if not self.intent_path.exists():
            raise FileNotFoundError(f"INTENT_PATH not found: {self.intent_path}")

        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        raw_id2label = config_data.get("id2label")
        if not raw_id2label:
            raise ValueError("id2label not found in model config.json")
        self.id2label = {int(k): v for k, v in raw_id2label.items()}

        with self.intent_path.open("r", encoding="utf-8") as f:
            intent_data = json.load(f)
        self.intent_reverse = intent_data.get("reverse", {}).get("level2", {})

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(self.onnx_path), providers=preferred)
        except Exception:
            self.session = ort.InferenceSession(
                str(self.onnx_path), providers=["CPUExecutionProvider"]
            )
        self.providers = self.session.get_providers()

    def predict(self, query: str) -> Dict[str, Any]:
        if self.session is None or self.tokenizer is None:
            raise RuntimeError("Model artifacts are not loaded.")

        enc = self.tokenizer(
            [query],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(["logits"], ort_inputs)[0]
        logits = np.asarray(logits)[0]

        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        pred_idx = int(np.argmax(probs))
        score = float(probs[pred_idx])

        intent_id = self.id2label[pred_idx]
        is_other = score < self.other_threshold

        level1 = None
        level2 = None
        if not is_other:
            mapped = self.intent_reverse.get(intent_id, {})
            level1 = mapped.get("level1")
            level2 = mapped.get("level2")
        else:
            intent_id = "OTHER"

        return {
            "intent_id": intent_id,
            "level1": level1,
            "level2": level2,
            "score": round(score, 6),
            "is_other": is_other,
            "model_version": self.model_version,
        }


artifacts = ModelArtifacts()
app = FastAPI(title="Intent Inference API", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    artifacts.load()


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_dir": str(artifacts.model_dir),
        "onnx_path": str(artifacts.onnx_path),
        "providers": artifacts.providers,
        "model_version": artifacts.model_version,
    }


@app.post("/v1/intent/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    result = artifacts.predict(query)
    return PredictResponse(**result)


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, workers=1)
