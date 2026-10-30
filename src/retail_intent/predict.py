import json
import math
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ID2LABEL, LABEL2ID, MAX_BATCH_SIZE, MAX_TEXT_CHARS


def validate_texts(texts: list[str]) -> list[str]:
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH_SIZE:
        raise ValueError(f"texts deve conter entre 1 e {MAX_BATCH_SIZE} mensagens.")
    if any(not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_TEXT_CHARS for text in texts):
        raise ValueError(f"Cada mensagem deve conter 1 a {MAX_TEXT_CHARS} caracteres.")
    return [text.strip() for text in texts]


class IntentClassifier:
    def __init__(self, model_dir: Path, device="cpu", threshold=0.75, allow_smoke=False):
        path = Path(model_dir)
        if not (path / "manifest.json").is_file():
            raise ValueError("Modelo não exportado: manifest.json ausente. Execute o treinamento primeiro.")
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("status") != "ready":
            raise ValueError("Manifesto incompatível ou treinamento incompleto.")
        if self.manifest.get("purpose") not in {"fine_tuning", "smoke"}:
            raise ValueError("Propósito do artefato inválido.")
        if self.manifest["purpose"] == "smoke" and not allow_smoke:
            raise ValueError("Modelo de ensaio bloqueado. ALLOW_SMOKE_MODEL=true somente para demonstração local.")
        if self.manifest.get("label2id") != LABEL2ID:
            raise ValueError("Mapa de classes do manifesto incompatível.")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("O limiar deve estar entre 0 e 1.")
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device deve ser cpu, cuda ou auto.")
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
        selected = "cpu" if selected == "auto" else selected
        if selected == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA indisponível.")
        self.device = torch.device(selected)
        self.threshold = threshold
        self.max_length = self.manifest.get("max_length")
        if type(self.max_length) is not int or not 8 <= self.max_length <= 512:
            raise ValueError("max_length do manifesto inválido.")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        )
        if self.model.config.label2id != LABEL2ID or self.model.config.num_labels != 2:
            raise ValueError("As classes do modelo não correspondem ao manifesto.")
        if self.max_length > self.model.config.max_position_embeddings:
            raise ValueError("Comprimento incompatível com o modelo.")
        self.model.to(self.device).eval()
        self._lock = Lock()

    def predict(self, texts: list[str]) -> list[dict]:
        """Calcula probabilidades sem gradientes; serializa o acesso ao modelo."""
        texts = validate_texts(texts)
        with self._lock, torch.inference_mode():
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
            ).to(self.device)
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().tolist()
        results = []
        for values in probabilities:
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("O modelo produziu probabilidades não finitas.")
            index = max(range(2), key=values.__getitem__)
            results.append({
                "label": ID2LABEL[index], "confidence": values[index],
                "probabilities": {ID2LABEL[i]: value for i, value in enumerate(values)},
                "needs_review": values[index] < self.threshold,
            })
        return results
