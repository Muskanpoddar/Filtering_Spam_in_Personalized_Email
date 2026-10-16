"""Project configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
PROFILES_DIR = ARTIFACTS_DIR / "profiles"

for _p in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR,
           RESULTS_DIR, PROFILES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
RANDOM_STATE = 42

SPAMASSASSIN_URL = "https://spamassassin.apache.org/old/publiccorpus/"
ENRON_SPAM_URL = "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/enron_spam.tar.gz"

LABEL_HAM = 0
LABEL_SPAM = 1