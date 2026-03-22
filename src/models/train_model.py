import os
import pickle

import pandas as pd
from ruamel.yaml import YAML
from sklearn.ensemble import RandomForestClassifier


def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")


def load_params(filepath: str) -> dict:
    try:
        yaml = YAML(typ='safe')
        with open(filepath) as f:
            params = yaml.load(f)
        return params.get("model_building", {})
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")


def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        raise RuntimeError(f"Error training model: {e}")


def save_model(model, output_path: str):
    try:
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"Error saving model to {output_path}: {e}")


def main():
    processed_data_path = os.path.join("data", "processed", "train_processed.csv")
    param_file_path = "params.yaml"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_output_path = os.path.join(models_dir, "model.pkl")

    data = load_data(processed_data_path)
    if "Mitigation_Action_Taken" not in data.columns:
        raise ValueError("Target column 'Mitigation_Action_Taken' not found in training data")
 
    X_train = data.drop("Mitigation_Action_Taken", axis=1)
    y_train = data["Mitigation_Action_Taken"]

    params = load_params(param_file_path)
    n_estimators = params.get("n_estimators", 100)

    model = train_model(X_train, y_train, n_estimators)
    save_model(model, model_output_path)


if __name__ == "__main__":
    main()