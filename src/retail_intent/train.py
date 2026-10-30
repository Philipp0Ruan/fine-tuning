import json
import math
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import ID2LABEL, LABEL2ID
from .data import prepare_data
from .metrics import classification_metrics, trainer_metrics


@dataclass(frozen=True)
class TrainConfig:
    train_file: Path
    test_file: Path
    output_dir: Path
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    revision: str = "main"
    epochs: float = 5
    batch_size: int = 8
    learning_rate: float = 2e-5
    validation_size: float = 0.2
    max_length: int = 128
    seed: int = 42
    device: str = "auto"
    purpose: str = "fine_tuning"

    def __post_init__(self):
        if not math.isfinite(self.epochs) or self.epochs <= 0:
            raise ValueError("epochs deve ser positivo e finito.")
        if not 1 <= self.batch_size <= 1024:
            raise ValueError("batch_size deve estar entre 1 e 1024.")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate deve estar entre 0 e 1.")
        if not 8 <= self.max_length <= 512:
            raise ValueError("max_length deve estar entre 8 e 512.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device deve ser auto, cpu ou cuda.")
        if self.purpose not in {"fine_tuning", "smoke"}:
            raise ValueError("purpose inválido.")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def run_training(config: TrainConfig) -> dict:
    """Treina com validação separada, exporta o melhor modelo e avalia o teste reservado."""
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output_dir precisa estar vazio ou não existir.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA solicitada, mas nenhuma GPU CUDA está disponível.")
    train, validation, test, audit = prepare_data(
        config.train_file, config.test_file, config.validation_size, config.seed
    )
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, revision=config.revision, num_labels=2,
        label2id=LABEL2ID, id2label=ID2LABEL, trust_remote_code=False,
        weights_only=True,
    )
    if config.max_length > model.config.max_position_embeddings:
        raise ValueError("max_length ultrapassa o limite de posições do modelo.")
    resolved_revision = getattr(model.config, "_commit_hash", None)

    def tokenized(rows):
        dataset = Dataset.from_dict({"text": [row.text for row in rows], "labels": [row.label for row in rows]})
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=config.max_length),
            batched=True, remove_columns=["text"],
        )

    train_dataset, validation_dataset = tokenized(train), tokenized(validation)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_audit.json", audit)
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        save_total_limit=2, report_to="none", seed=config.seed, data_seed=config.seed,
        use_cpu=config.device == "cpu", dataloader_num_workers=0,
        dataloader_pin_memory=config.device != "cpu" and torch.cuda.is_available(),
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=validation_dataset,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_metrics,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)

    validation_prediction = trainer.predict(validation_dataset)
    test_prediction = trainer.predict(tokenized(test))
    baseline = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.seed),
    )
    baseline.fit([row.text for row in train], [row.label for row in train])
    majority = Counter(row.label for row in train).most_common(1)[0][0]
    metrics = {
        "purpose": config.purpose,
        "best_validation_f1_macro": trainer.state.best_metric,
        "validation": classification_metrics(validation_prediction.label_ids, np.argmax(validation_prediction.predictions, axis=-1)),
        "test": classification_metrics(test_prediction.label_ids, np.argmax(test_prediction.predictions, axis=-1)),
        "baseline_tfidf": {
            name: classification_metrics([row.label for row in rows], baseline.predict([row.text for row in rows]))
            for name, rows in (("validation", validation), ("test", test))
        },
        "baseline_majority_test": classification_metrics([row.label for row in test], [majority] * len(test)),
    }
    write_json(output / "metrics.json", metrics)
    manifest = {
        "schema_version": 1, "status": "ready", "purpose": config.purpose,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": config.model_name, "requested_revision": config.revision,
        "resolved_revision": resolved_revision,
        "label2id": LABEL2ID, "max_length": config.max_length,
        "training": asdict(config), "data": audit,
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "device": str(trainer.args.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {name: version(name) for name in (
                "torch", "transformers", "datasets", "accelerate", "numpy", "scikit-learn"
            )},
        },
    }
    write_json(output / "manifest.json", manifest)
    return metrics
