import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

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


def load_params(params_path: str) -> float:
    logger.debug("Loading params from %s", params_path)
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        test_size = params["data_ingestion"]["test_size"]
        logger.debug("Loaded test_size=%.2f", test_size)
        return test_size
    except FileNotFoundError:
        raise FileNotFoundError(f"Params file not found: {params_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in params.yaml: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse params.yaml: {e}")


def load_data(url: str) -> pd.DataFrame:
    logger.debug("Loading data from %s", url)
    try:
        df = pd.read_csv(url)
        logger.debug("Loaded DataFrame with shape %s", df.shape)
        if "tweet_id" in df.columns:
            df.drop(columns=["tweet_id"], inplace=True)
            logger.debug("Dropped 'tweet_id' column")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {url}: {e}")


def filter_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Filtering and encoding sentiments, input shape=%s", df.shape)
    try:
        if "sentiment" not in df.columns:
            raise ValueError("Column 'sentiment' not found in DataFrame")
        filtered = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
        if filtered.empty:
            raise ValueError("No rows matched the sentiment filter ('happiness', 'sadness')")
        filtered["sentiment"].replace({"happiness": 1, "sadness": 0}, inplace=True)
        logger.debug("Filtered DataFrame shape=%s", filtered.shape)
        return filtered
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed during filter and encode step: {e}")


def split_data(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("Splitting data with test_size=%.2f", test_size)
    try:
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        train, test = train_test_split(df, test_size=test_size, random_state=42)
        logger.debug("Train shape=%s, Test shape=%s", train.shape, test.shape)
        return train, test
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to split data: {e}")


def save_data(train: pd.DataFrame, test: pd.DataFrame, output_dir: str) -> None:
    logger.debug("Saving data to %s", output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        logger.info("Data saved to %s", output_dir)
    except OSError as e:
        raise OSError(f"Failed to save data to {output_dir}: {e}")


def main():
    _setup_logging()
    DATA_URL = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
    PARAMS_PATH = "params.yaml"
    OUTPUT_DIR = os.path.join("data", "raw")

    try:
        logger.info("Starting data ingestion")
        test_size = load_params(PARAMS_PATH)
        df = load_data(DATA_URL)
        final_df = filter_and_encode(df)
        train_data, test_data = split_data(final_df, test_size)
        save_data(train_data, test_data, OUTPUT_DIR)
        logger.info("Data ingestion completed successfully.")
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError) as e:
        logger.error("Data ingestion failed: %s", e)
        raise


if __name__ == "__main__":
    main()
