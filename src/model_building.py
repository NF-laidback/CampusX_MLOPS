import pandas as pd
import os
import logging
import pickle
import xgboost as xgb
import yaml

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


def load_params(params_path: str = "params.yaml") -> dict:
    logger.debug("Loading params from %s", params_path)
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        model_params = params["model_building"]
        logger.debug("Loaded model_building params: %s", model_params)
        return model_params
    except FileNotFoundError:
        raise FileNotFoundError(f"Params file not found: {params_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in params.yaml: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse params.yaml: {e}")


def load_data(feature_dir: str) -> pd.DataFrame:
    logger.debug("Loading feature data from %s", feature_dir)
    try:
        train_path = os.path.join(feature_dir, "train_feature_engineered.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Feature file not found: {train_path}")
        df = pd.read_csv(train_path)
        logger.debug("Loaded feature DataFrame with shape %s", df.shape)
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load feature data from {feature_dir}: {e}")


def prepare_features(train_data: pd.DataFrame) -> tuple:
    logger.debug("Preparing training features")
    try:
        if "label" not in train_data.columns:
            raise ValueError("Column 'label' not found in train DataFrame")
        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"].map({"sadness": 0, "happiness": 1})
        logger.debug("X_train shape=%s, y_train shape=%s", X_train.shape, y_train.shape)
        return X_train, y_train
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to prepare features: {e}")


def train_model(X_train, y_train, params: dict) -> xgb.XGBClassifier:
    logger.debug("Training XGBClassifier with params: %s", params)
    try:
        for key in ("n_estimators", "learning_rate"):
            if key not in params:
                raise KeyError(f"Missing model param: '{key}'")
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
        )
        model.fit(X_train, y_train)
        logger.info("Model training complete")
        return model
    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}")


def save_model(model: xgb.XGBClassifier, output_dir: str) -> None:
    logger.debug("Saving model to %s", output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "xgb_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", model_path)
    except OSError as e:
        raise OSError(f"Failed to save model to {output_dir}: {e}")
    except pickle.PicklingError as e:
        raise RuntimeError(f"Failed to serialize model: {e}")


def main():
    _setup_logging()
    try:
        logger.info("Starting model building")
        params = load_params()
        train_data = load_data(os.path.join("data", "feature_engineered"))
        X_train, y_train = prepare_features(train_data)
        model = train_model(X_train, y_train, params)
        save_model(model, "models")
        logger.info("Model building completed successfully.")
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError) as e:
        logger.error("Model building failed: %s", e)
        raise


if __name__ == "__main__":
    main()
