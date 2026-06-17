import pandas as pd
import os
import pickle
import xgboost as xgb
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        return params["model_building"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Params file not found: {params_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in params.yaml: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse params.yaml: {e}")


def load_data(feature_dir: str) -> pd.DataFrame:
    try:
        train_path = os.path.join(feature_dir, "train_feature_engineered.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Feature file not found: {train_path}")
        return pd.read_csv(train_path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load feature data from {feature_dir}: {e}")


def prepare_features(train_data: pd.DataFrame) -> tuple:
    try:
        if "label" not in train_data.columns:
            raise ValueError("Column 'label' not found in train DataFrame")
        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"].map({"sadness": 0, "happiness": 1})
        return X_train, y_train
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to prepare features: {e}")


def train_model(X_train, y_train, params: dict) -> xgb.XGBClassifier:
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
        return model
    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}")


def save_model(model: xgb.XGBClassifier, output_dir: str) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump(model, f)
    except OSError as e:
        raise OSError(f"Failed to save model to {output_dir}: {e}")
    except pickle.PicklingError as e:
        raise RuntimeError(f"Failed to serialize model: {e}")


def main():
    try:
        params = load_params()
        train_data = load_data(os.path.join("data", "feature_engineered"))
        X_train, y_train = prepare_features(train_data)
        model = train_model(X_train, y_train, params)
        save_model(model, "models")
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError):
        raise


if __name__ == "__main__":
    main()
