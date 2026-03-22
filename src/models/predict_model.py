import json
import os

import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_test_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading test data from {file_path}: {e}")


def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {e}")


def evaluate_model(model, X_test, y_test) -> dict:
    try:
        y_pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
    except Exception as e:
        raise RuntimeError(f"Error evaluating model: {e}")


def save_metrics(metrics: dict, output_path: str):
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {output_path}: {e}")


def main():
    test_data_path = r"data/processed/test_processed.csv"
    model_path = os.path.join("models", "model.pkl")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = os.path.join(reports_dir, "metrics.json")

    test_data = load_test_data(test_data_path)
    if 'Mitigation_Action_Taken' not in test_data.columns:
        raise ValueError("Target column 'Mitigation_Action_Taken' not found in test data")

    X_test = test_data.drop('Mitigation_Action_Taken', axis=1)
    y_test = test_data['Mitigation_Action_Taken']

    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_path)


if __name__ == "__main__":
    main()