import numpy as np
import pandas as pd
import os
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def load_model(model_path: str):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise
    except pickle.UnpicklingError as e:
        raise RuntimeError(f"Failed to deserialize model from {model_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def load_test_data(feature_dir: str) -> pd.DataFrame:
    try:
        test_path = os.path.join(feature_dir, "test_feature_engineered.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test feature file not found: {test_path}")
        return pd.read_csv(test_path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load test data from {feature_dir}: {e}")


def prepare_features(test_data: pd.DataFrame) -> tuple:
    try:
        if "label" not in test_data.columns:
            raise ValueError("Column 'label' not found in test DataFrame")
        X_test = test_data.drop(columns=["label"])
        y_test = test_data["label"].map({"sadness": 0, "happiness": 1})
        return X_test, y_test
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to prepare test features: {e}")


def evaluate_model(model, X_test, y_test) -> dict:
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }
    except AttributeError as e:
        raise RuntimeError(f"Model does not support required prediction methods: {e}")
    except Exception as e:
        raise RuntimeError(f"Model evaluation failed: {e}")


def save_metrics(metrics: dict, output_path: str) -> None:
    try:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
    except OSError as e:
        raise OSError(f"Failed to save metrics to {output_path}: {e}")
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Failed to serialize metrics to JSON: {e}")


def main():
    try:
        model = load_model(os.path.join("models", "xgb_model.pkl"))
        test_data = load_test_data(os.path.join("data", "feature_engineered"))
        X_test, y_test = prepare_features(test_data)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "metrics.json")
    except (FileNotFoundError, ValueError, RuntimeError, OSError):
        raise


if __name__ == "__main__":
    main()
