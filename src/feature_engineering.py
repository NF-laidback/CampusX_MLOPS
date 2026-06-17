import numpy as np
import pandas as pd
import os
import yaml
from sklearn.feature_extraction.text import CountVectorizer


def load_params(params_path: str = "params.yaml") -> int:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        return params["feature_engineering"]["max_features"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Params file not found: {params_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in params.yaml: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse params.yaml: {e}")


def load_data(processed_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_path = os.path.join(processed_dir, "train_processed.csv")
        test_path = os.path.join(processed_dir, "test_processed.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Processed train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Processed test file not found: {test_path}")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load processed data from {processed_dir}: {e}")


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
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
    try:
        if not isinstance(max_features, int) or max_features <= 0:
            raise ValueError(f"max_features must be a positive integer, got {max_features}")
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        return X_train_bow, X_test_bow
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to apply Bag-of-Words vectorization: {e}")


def build_dataframes(X_train_bow, y_train, X_test_bow, y_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test
        return train_df, test_df
    except Exception as e:
        raise RuntimeError(f"Failed to build feature DataFrames: {e}")


def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_feature_engineered.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_feature_engineered.csv"), index=False)
    except OSError as e:
        raise OSError(f"Failed to save feature files to {output_dir}: {e}")


def main():
    try:
        max_features = load_params()
        train_data, test_data = load_data(os.path.join("data", "processed"))
        X_train, y_train, X_test, y_test = prepare_features(train_data, test_data)
        X_train_bow, X_test_bow = apply_bow(X_train, X_test, max_features)
        train_df, test_df = build_dataframes(X_train_bow, y_train, X_test_bow, y_test)
        save_features(train_df, test_df, os.path.join("data", "feature_engineered"))
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError):
        raise


if __name__ == "__main__":
    main()
