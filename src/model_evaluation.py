import numpy as np
import pandas as pd
import os
import json
import logging
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if not root_logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join("logs", "pipeline.log"))
        fh.setFormatter(fmt)
        root_logger.addHandler(sh)
        root_logger.addHandler(fh)


def load_model(model_path: str):
    logger.debug("Loading model from %s", model_path)
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully")
        return model
    except FileNotFoundError:
        raise
    except pickle.UnpicklingError as e:
        raise RuntimeError(f"Failed to deserialize model from {model_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def load_test_data(feature_dir: str) -> pd.DataFrame:
    logger.debug("Loading test data from %s", feature_dir)
    try:
        test_path = os.path.join(feature_dir, "test_feature_engineered.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test feature file not found: {test_path}")
        df = pd.read_csv(test_path)
        logger.debug("Loaded test DataFrame with shape %s", df.shape)
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load test data from {feature_dir}: {e}")


def prepare_features(test_data: pd.DataFrame) -> tuple:
    logger.debug("Preparing test features")
    try:
        if "label" not in test_data.columns:
            raise ValueError("Column 'label' not found in test DataFrame")
        X_test = test_data.drop(columns=["label"])
        y_test = test_data["label"].map({"sadness": 0, "happiness": 1})
        logger.debug("X_test shape=%s, y_test shape=%s", X_test.shape, y_test.shape)
        return X_test, y_test
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to prepare test features: {e}")


def evaluate_model(model, X_test, y_test) -> dict:
    logger.debug("Evaluating model on test set")
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics
    except AttributeError as e:
        raise RuntimeError(f"Model does not support required prediction methods: {e}")
    except Exception as e:
        raise RuntimeError(f"Model evaluation failed: {e}")


def save_metrics(metrics: dict, output_path: str) -> None:
    logger.debug("Saving metrics to %s", output_path)
    try:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics saved to %s", output_path)
    except OSError as e:
        raise OSError(f"Failed to save metrics to {output_path}: {e}")
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Failed to serialize metrics to JSON: {e}")


def main():
    _setup_logging()
    try:
        logger.info("Starting model evaluation")
        model = load_model(os.path.join("models", "xgb_model.pkl"))
        test_data = load_test_data(os.path.join("data", "feature_engineered"))
        X_test, y_test = prepare_features(test_data)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "metrics.json")
        logger.info("Model evaluation completed successfully.")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        logger.error("Model evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()
