import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(labels, predictions) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "label_order": ["suporte", "venda"],
        "per_class": classification_report(
            labels, predictions, labels=[0, 1], target_names=["suporte", "venda"],
            output_dict=True, zero_division=0,
        ),
    }


def trainer_metrics(evaluation) -> dict:
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
    }
