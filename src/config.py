"""
Central configuration: file paths, URLs, model hyperparameters, constants.
No secrets here — API keys go in .env (see .env.example).
"""
import os
from pathlib import Path

# Paths
SRC_DIR     = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR    = PROJECT_DIR / "data"
DOCS_DIR    = PROJECT_DIR / "docs"

# Local CSV data files
CRC_DATASET_CSV               = DATA_DIR / "crc_dataset.csv"
COLORECTAL_CANCER_DATASET_CSV = DATA_DIR / "colorectal_cancer_dataset.csv"

# Output artifacts
TRIALS_OUTPUT_JSON = DATA_DIR / "colon_cancer_trials_survival.json"

# API endpoints
NCI_TRIALS_BASE_URL = "https://clinicaltrialsapi.cancer.gov/api/v2"
CDC_WONDER_URL      = "https://wonder.cdc.gov/controller/datarequest/D207"

# API behavior
TARGET_TRIALS    = 400
TRIALS_PAGE_SIZE = 50
HTTP_TIMEOUT     = 120

# ML hyperparameters
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_SPLITS    = 5

# Environment variable names
ENV_NCI_API_KEY = "NCI_API_KEY"

# Read the NCI Clinical Trials API key from the environment
def get_nci_api_key() -> str:
    key = os.getenv(ENV_NCI_API_KEY)
    if not key:
        raise RuntimeError(
            f"{ENV_NCI_API_KEY} not set. "
            f"Copy .env.example to .env and fill in your key, "
            f"or export {ENV_NCI_API_KEY} in your shell."
        )
    return key
