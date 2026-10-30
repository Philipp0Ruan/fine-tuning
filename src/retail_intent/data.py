import hashlib
import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from .config import LABEL2ID, MAX_TEXT_CHARS


@dataclass(frozen=True)
class Sample:
    text: str
    label: int


def normalized_text(text: str) -> str:
    """Normaliza apenas para detectar duplicação, sem alterar o texto do modelo."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path: Path) -> tuple[list[Sample], dict]:
    path = Path(path)
    samples, seen = [], {}
    duplicates = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: JSON inválido.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_no}: esperado um objeto JSON.")
            text, label = row.get("prompt"), row.get("completion")
            if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_TEXT_CHARS:
                raise ValueError(f"{path.name}:{line_no}: prompt deve ter 1 a {MAX_TEXT_CHARS} caracteres.")
            if not isinstance(label, str) or label not in LABEL2ID:
                raise ValueError(f"{path.name}:{line_no}: completion deve ser suporte ou venda.")
            key = normalized_text(text)
            if not key:
                raise ValueError(f"{path.name}:{line_no}: prompt vazio após normalização.")
            if key in seen:
                if seen[key] != LABEL2ID[label]:
                    raise ValueError(f"{path.name}:{line_no}: rótulos contraditórios para o mesmo texto.")
                duplicates += 1
                continue
            seen[key] = LABEL2ID[label]
            samples.append(Sample(text.strip(), LABEL2ID[label]))
    counts = Counter(sample.label for sample in samples)
    if set(counts) != {0, 1}:
        raise ValueError(f"{path.name}: são necessárias amostras únicas das duas classes.")
    return samples, {
        "file_name": path.name,
        "sha256": file_hash(path),
        "unique_rows": len(samples),
        "duplicates_removed": duplicates,
        "class_counts": {label: counts[index] for label, index in LABEL2ID.items()},
    }


def prepare_data(train_file: Path, test_file: Path, validation_size=0.2, seed=42):
    """Reserva validação do treino e rejeita interseção com o teste final."""
    if not math.isfinite(validation_size) or not 0 < validation_size < 1:
        raise ValueError("validation_size deve estar entre 0 e 1, sem os extremos.")
    original, train_audit = read_jsonl(train_file)
    test, test_audit = read_jsonl(test_file)
    overlap = {normalized_text(row.text) for row in original} & {
        normalized_text(row.text) for row in test
    }
    if overlap:
        raise ValueError(f"Vazamento: {len(overlap)} textos aparecem em treino e teste.")
    if min(Counter(row.label for row in original).values()) < 2:
        raise ValueError("O treino precisa de pelo menos duas amostras únicas por classe.")
    try:
        train, validation = train_test_split(
            original,
            test_size=validation_size,
            random_state=seed,
            stratify=[row.label for row in original],
        )
    except ValueError as exc:
        raise ValueError("Dados insuficientes para a divisão estratificada; aumente o conjunto ou ajuste validation_size.") from exc
    if any({row.label for row in split} != {0, 1} for split in (train, validation)):
        raise ValueError("A divisão deve manter ambas as classes em treino e validação.")
    audit = {
        "train_source": train_audit,
        "test_source": test_audit,
        "split_seed": seed,
        "validation_size": validation_size,
        "split_counts": {name: len(rows) for name, rows in (
            ("train", train), ("validation", validation), ("test", test)
        )},
        "overlap_count": 0,
    }
    return train, validation, test, audit
