import numpy as np
import pandas as pd
import os
import logging
import yaml
from sklearn.feature_extraction.text import CountVectorizer

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


def load_params(params_path: str = "params.yaml") -> int:
    logger.debug("Loading params from %s", params_path)
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        max_features = params["feature_engineering"]["max_features"]
        logger.debug("Loaded max_features=%d", max_features)
        return max_features
    except FileNotFoundError:
        raise FileNotFoundError(f"Params file not found: {params_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in params.yaml: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse params.yaml: {e}")


def load_data(processed_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("Loading processed data from %s", processed_dir)
    try:
        train_path = os.path.join(processed_dir, "train_processed.csv")
        test_path = os.path.join(processed_dir, "test_processed.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Processed train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Processed test file not found: {test_path}")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        logger.debug("Loaded train shape=%s, test shape=%s", train.shape, test.shape)
        return train, test
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load processed data from {processed_dir}: {e}")


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    logger.debug("Preparing features from DataFrames")
    try:
        for col in ("content", "sentiment"):
            if col not in train.columns:
                raise ValueError(f"Column '{col}' not found in train DataFrame")
            if col not in test.columns:
                raise ValueError(f"Column '{col}' not found in test DataFrame")
        train = train.dropna(subset=["content"])
        test = test.dropna(subset=["content"])
        if train.empty:
            raise ValueError("Train DataFrame is empty after dropping NaN content rows")
        if test.empty:
            raise ValueError("Test DataFrame is empty after dropping NaN content rows")
        logger.debug("Train samples=%d, Test samples=%d", len(train), len(test))
        return (
            train["content"].values,
            train["sentiment"].values,
            test["content"].values,
            test["sentiment"].values,
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to prepare features: {e}")


def apply_bow(X_train, X_test, max_features: int) -> tuple:
    logger.debug("Applying Bag-of-Words with max_features=%d", max_features)
    try:
        if not isinstance(max_features, int) or max_features <= 0:
            raise ValueError(f"max_features must be a positive integer, got {max_features}")
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logger.debug("BoW train shape=%s, test shape=%s", X_train_bow.shape, X_test_bow.shape)
        return X_train_bow, X_test_bow
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to apply Bag-of-Words vectorization: {e}")


def build_dataframes(X_train_bow, y_train, X_test_bow, y_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("Building feature DataFrames")
    try:
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test
        logger.debug("Feature train shape=%s, test shape=%s", train_df.shape, test_df.shape)
        return train_df, test_df
    except Exception as e:
        raise RuntimeError(f"Failed to build feature DataFrames: {e}")


def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    logger.debug("Saving feature files to %s", output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_feature_engineered.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_feature_engineered.csv"), index=False)
        logger.info("Feature files saved to %s", output_dir)
    except OSError as e:
        raise OSError(f"Failed to save feature files to {output_dir}: {e}")


def main():
    _setup_logging()
    try:
        logger.info("Starting feature engineering")
        max_features = load_params()
        train_data, test_data = load_data(os.path.join("data", "processed"))
        X_train, y_train, X_test, y_test = prepare_features(train_data, test_data)
        X_train_bow, X_test_bow = apply_bow(X_train, X_test, max_features)
        train_df, test_df = build_dataframes(X_train_bow, y_train, X_test_bow, y_test)
        save_features(train_df, test_df, os.path.join("data", "feature_engineered"))
        logger.info("Feature engineering completed successfully.")
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError) as e:
        logger.error("Feature engineering failed: %s", e)
        raise


if __name__ == "__main__":
    main()
