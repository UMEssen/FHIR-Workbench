import json
import numpy as np
import logging
from sklearn.metrics import f1_score
import re

logging.basicConfig(level=logging.INFO)

def micro_f1_score(items: list[tuple]) -> float:
    golds, preds = zip(*items)
    return f1_score(golds, preds, average="micro")

def extract_json_from_text(text: str) -> str:
    match = re.search(r'```json\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError("No JSON block found in the text.")

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(flatten_dict({f"{k}[{i}]": item}, parent_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_fhir_f1_score(ground_truth: dict, prediction: dict) -> float:
    true_set = set(flatten_dict(ground_truth).items())
    pred_set = set(flatten_dict(prediction).items())

    tp, fp, fn = len(true_set & pred_set), len(pred_set - true_set), len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

def fhir_f1_score(references: list[str], predictions: list[str], **kwargs) -> float:
    scores, errors = [], []

    for ref_str, pred_str in zip(references, predictions):
        try:
            pred_json = extract_json_from_text(pred_str)
            fhir_true = json.loads(ref_str)
            fhir_pred = json.loads(pred_json)
            scores.append(calculate_fhir_f1_score(fhir_true, fhir_pred))
        except Exception as e:
            scores.append(0.0)
            errors.append(f"Error processing prediction: {e}")

    if errors:
        logging.warning("Errors encountered during scoring:")
        for error in errors:
            logging.warning(error)

    return np.mean(scores) if scores else 0.0
