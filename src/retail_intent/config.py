import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LABEL2ID = {"suporte": 0, "venda": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
MAX_TEXT_CHARS = 4000
MAX_BATCH_SIZE = 32


@dataclass(frozen=True)
class Settings:
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    model_revision: str = "main"
    model_dir: Path = Path("artifacts/model")
    max_length: int = 128
    confidence_threshold: float = 0.75
    device: str = "cpu"
    allow_smoke_model: bool = False
    python_host: str = "127.0.0.1"
    python_port: int = 8000

    def __post_init__(self):
        if not 8 <= self.max_length <= 512:
            raise ValueError("MAX_LENGTH deve estar entre 8 e 512.")
        if not math.isfinite(self.confidence_threshold) or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD deve estar entre 0 e 1.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("DEVICE deve ser cpu, cuda ou auto.")
        if not 1 <= self.python_port <= 65535:
            raise ValueError("PYTHON_PORT deve estar entre 1 e 65535.")

    @classmethod
    def from_env(cls):
        load_dotenv(Path.cwd() / ".env", override=False)
        flag = os.getenv("ALLOW_SMOKE_MODEL", "false").lower()
        if flag not in {"true", "false"}:
            raise ValueError("ALLOW_SMOKE_MODEL deve ser true ou false.")
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            model_revision=os.getenv("MODEL_REVISION", "main"),
            model_dir=Path(os.getenv("MODEL_DIR", "artifacts/model")),
            max_length=int(os.getenv("MAX_LENGTH", "128")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")),
            device=os.getenv("DEVICE", "cpu"),
            allow_smoke_model=flag == "true",
            python_host=os.getenv("PYTHON_HOST", "127.0.0.1"),
            python_port=int(os.getenv("PYTHON_PORT", "8000")),
        )
