import numpy as np
import pandas as pd
import os
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


def download_nltk_resources():
    logger.debug("Downloading NLTK resources")
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.debug("NLTK resources downloaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to download NLTK resources: {e}")


def load_data(raw_data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("Loading raw data from %s", raw_data_path)
    try:
        train_path = os.path.join(raw_data_path, "train.csv")
        test_path = os.path.join(raw_data_path, "test.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug("Loaded train shape=%s, test shape=%s", train_data.shape, test_data.shape)
        return train_data, test_data
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {raw_data_path}: {e}")


def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        return " ".join(lemmatizer.lemmatize(word) for word in str(text).split())
    except Exception:
        return text


def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        return " ".join(word for word in str(text).split() if word not in stop_words)
    except Exception:
        return text


def removing_numbers(text: str) -> str:
    try:
        return "".join(ch for ch in str(text) if not ch.isdigit())
    except Exception:
        return text


def lower_case(text: str) -> str:
    try:
        return " ".join(word.lower() for word in str(text).split())
    except Exception:
        return text


def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', str(text))
        text = text.replace('؛', "")
        text = re.sub(r'\s+', ' ', text)
        return " ".join(text.split()).strip()
    except Exception:
        return text


def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', str(text))
    except Exception:
        return text


def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Removing small sentences, input shape=%s", df.shape)
    try:
        if 'text' not in df.columns:
            raise ValueError("Column 'text' not found in DataFrame")
        mask = df['text'].apply(lambda t: len(str(t).split()) < 3)
        df.loc[mask, 'text'] = np.nan
        logger.debug("Removed %d small sentences", mask.sum())
        return df
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to remove small sentences: {e}")


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Normalizing text, input shape=%s", df.shape)
    try:
        if 'content' not in df.columns:
            raise ValueError("Column 'content' not found in DataFrame")
        pipeline = [
            lower_case,
            remove_stop_words,
            removing_numbers,
            removing_punctuations,
            removing_urls,
            lemmatization,
        ]
        for step in pipeline:
            logger.debug("Applying text step: %s", step.__name__)
            df['content'] = df['content'].apply(step)
        logger.debug("Text normalization complete, output shape=%s", df.shape)
        return df
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Text normalization failed: {e}")


def save_data(train: pd.DataFrame, test: pd.DataFrame, processed_data_path: str) -> None:
    logger.debug("Saving processed data to %s", processed_data_path)
    try:
        os.makedirs(processed_data_path, exist_ok=True)
        train.to_csv(os.path.join(processed_data_path, "train_processed.csv"), index=False)
        test.to_csv(os.path.join(processed_data_path, "test_processed.csv"), index=False)
        logger.info("Processed data saved to %s", processed_data_path)
    except OSError as e:
        raise OSError(f"Failed to save processed data to {processed_data_path}: {e}")


def main():
    _setup_logging()
    raw_data_path = os.path.join("data", "raw")
    processed_data_path = os.path.join("data", "processed")

    try:
        logger.info("Starting data preprocessing")
        download_nltk_resources()
        train_data, test_data = load_data(raw_data_path)
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)
        save_data(train_processed, test_processed, processed_data_path)
        logger.info("Data preprocessing completed successfully.")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        logger.error("Data preprocessing failed: %s", e)
        raise


if __name__ == "__main__":
    main()
