from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# Specific paths
DB_PATH = DATA_DIR / "indicators.db"
INDEX_PATH = DATA_DIR / "indicators_index"
CSV_PATH = DATA_DIR / "cleaned_indicators.csv"

# Model configuration
MODEL_CONFIG = {
    "model": "llama3-70b-8192",
    "temperature": 0.1,
    "api_key": os.getenv("GROQ_API_KEY")
}

# Database configuration
DB_URI = f"sqlite:///{DB_PATH}"