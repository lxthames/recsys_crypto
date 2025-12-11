from pathlib import Path

# Project root = .../recsys_crypto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = str(PROJECT_ROOT / "data")

# SPMF jar path (if/when you use it)
SPMF_JAR_PATH = str(PROJECT_ROOT / "tools" / "spmf.jar")

# Default Ollama model (change as you like: "llama3", "gemma2:9b", "phi3", etc.)
OLLAMA_MODEL_DEFAULT = "mistral"

# Exchange + assets settings
ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
]

EXCHANGE = "binance"
TIMEFRAME = "1h"
WINDOW_DAYS = 45          # ~45 days history
HORIZON_HOURS = 72        # 3 days -> 72 hours
