import argparse
import json
from pathlib import Path

from .config import Settings


def main():
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Classificação de mensagens em suporte e venda.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "train"):
        command = commands.add_parser(name)
        command.add_argument("--train-file", type=Path, required=True)
        command.add_argument("--test-file", type=Path, required=True)
        command.add_argument("--validation-size", type=float, default=0.2)
        command.add_argument("--seed", type=int, default=42)
        if name == "train":
            command.add_argument("--output-dir", type=Path, default=settings.model_dir)
            command.add_argument("--model-name", default=settings.model_name)
            command.add_argument("--revision", default=settings.model_revision)
            command.add_argument("--epochs", type=float, default=5)
            command.add_argument("--batch-size", type=int, default=8)
            command.add_argument("--learning-rate", type=float, default=2e-5)
            command.add_argument("--max-length", type=int, default=settings.max_length)
            command.add_argument("--device", choices=["auto", "cpu", "cuda"], default=settings.device)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--model-dir", type=Path, default=settings.model_dir)
    prediction.add_argument("--text", action="append", required=True, help="Repita a opção para enviar várias mensagens.")
    prediction.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            from .data import prepare_data

            *_, result = prepare_data(args.train_file, args.test_file, args.validation_size, args.seed)
        elif args.command == "train":
            from .train import TrainConfig, run_training

            options = vars(args).copy()
            options.pop("command")
            result = run_training(TrainConfig(**options))
        else:
            from .predict import IntentClassifier

            classifier = IntentClassifier(args.model_dir, settings.device, settings.confidence_threshold, args.allow_smoke)
            result = {"predictions": classifier.predict(args.text)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
