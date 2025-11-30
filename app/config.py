from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
LOGS_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database path
DB_PATH = DATA_DIR / "app.db"

# JWT & auth settings
SECRET_KEY = "CHANGE_THIS_SECRET_IN_REAL_PROJECT"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Token pricing
TRAIN_TOKENS_COST = 1
TRAIN_MULTI_TOKENS_COST = 1
PREDICT_TOKENS_COST = 5

