"""
Configuration module for Autonomous Quantitative AI Trading Agent.
Loads environment variables and defines quantitative strategy and risk parameters.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if it exists
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# Alpaca API Credentials
# ==============================================================================
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "True").lower() in ("true", "1", "t", "yes")

# Base URLs for Alpaca Trading and Data
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"


def has_valid_alpaca_credentials() -> bool:
    """Returns True if user has configured non-placeholder Alpaca credentials."""
    return bool(
        ALPACA_API_KEY
        and ALPACA_SECRET_KEY
        and not ALPACA_API_KEY.startswith("your_")
        and not ALPACA_SECRET_KEY.startswith("your_")
    )


# ==============================================================================
# Strategy & Asset Parameters
# ==============================================================================
# Default target asset symbol
SYMBOL: str = os.getenv("SYMBOL", "BTC-USD")

# Multi-asset crypto universe
SYMBOLS: list = ["BTC-USD", "ETH-USD", "SOL-USD"]

# Timeframe for bars (e.g., '1Hour', '15Min', '1Day')
TIMEFRAME: str = os.getenv("TIMEFRAME", "1Hour")

# Historical data lookback period for training (in days)
LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "730"))

# Model decision threshold: Win probability required to trigger BUY (Calibrated optimal 0.54)
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.54"))

# ==============================================================================
# Triple-Barrier & Volatility Parameters
# ==============================================================================
# Stop Loss distance as a multiple of Average True Range (ATR)
ATR_SL_MULTIPLIER: float = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))

# Take Profit distance as a multiple of Average True Range (ATR) -> 3.0 / 1.5 = 2:1 Reward-to-Risk
ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULTIPLIER", "3.0"))

# Maximum holding period (in bars/candles) before position evaluation expires in Triple-Barrier (24h)
HORIZON_BARS: int = int(os.getenv("HORIZON_BARS", "24"))

# ==============================================================================
# Risk Management & Portfolio Constraints
# ==============================================================================
# Maximum capital risk per individual trade (1.0% of total equity)
MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))

# Minimum risk floor for Kelly fraction scaling (0.25% of total equity)
MIN_RISK_PER_TRADE: float = 0.0025

# Maximum allowable exposure for a single position relative to buying power/equity (50%)
MAX_POSITION_EXPOSURE: float = float(os.getenv("MAX_POSITION_EXPOSURE", "0.50"))

# Circuit breaker threshold: Max allowable intra-day drawdown (3%) before halting trading
MAX_DAILY_DRAWDOWN: float = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.03"))

# Commission/slippage estimate per trade used in backtesting (0.15% per side for crypto)
DEFAULT_SLIPPAGE_BPS: float = float(os.getenv("DEFAULT_SLIPPAGE_BPS", "0.0015"))  # 15 bps (0.15%)

# ==============================================================================
# Sentiment Analysis & FinBERT Gatekeeper Parameters
# ==============================================================================
SENTIMENT_VETO_THRESHOLD: float = float(os.getenv("SENTIMENT_VETO_THRESHOLD", "-0.15"))
SENTIMENT_BOOST_THRESHOLD: float = float(os.getenv("SENTIMENT_BOOST_THRESHOLD", "0.25"))
SENTIMENT_BOOST_VALUE: float = float(os.getenv("SENTIMENT_BOOST_VALUE", "0.03"))

# ==============================================================================
# Self-Learning & Experience Replay Parameters
# ==============================================================================
RETRAIN_INTERVAL_DAYS: int = int(os.getenv("RETRAIN_INTERVAL_DAYS", "7"))
FAILURE_SAMPLE_WEIGHT: float = float(os.getenv("FAILURE_SAMPLE_WEIGHT", "3.0"))
SUCCESS_SAMPLE_WEIGHT: float = float(os.getenv("SUCCESS_SAMPLE_WEIGHT", "1.5"))
EXPERIENCE_REPLAY_WINDOW_HOURS: float = float(os.getenv("EXPERIENCE_REPLAY_WINDOW_HOURS", "2.0"))

# ==============================================================================
# Storage & Model Artifact Paths
# ==============================================================================
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgb_trading_model.joblib"
ACTIVE_CRYPTO_POSITIONS_FILE = MODEL_DIR / "active_crypto_positions.json"
LOCAL_ACCOUNT_FILE = MODEL_DIR / "local_paper_account.json"
TRADE_HISTORY_FILE = MODEL_DIR / "trade_history.csv"
RETRAIN_HISTORY_FILE = MODEL_DIR / "retrain_history.json"
DASHBOARD_FILE = MODEL_DIR / "dashboard.html"
TELEMETRY_FILE = MODEL_DIR / "latest_telemetry.json"
ROOT_DASHBOARD_FILE = BASE_DIR / "index.html"
ROOT_TELEMETRY_FILE = BASE_DIR / "latest_telemetry.json"
LOG_FILE = BASE_DIR / "trading_agent.log"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_model_path(symbol: str, for_training: bool = False) -> Path:
    """Returns symbol-specific model path or fallback model path."""
    clean_sym = symbol.replace("/", "-").upper()
    sym_path = MODEL_DIR / f"xgb_{clean_sym}.joblib"
    if for_training:
        return sym_path
    if sym_path.exists():
        return sym_path
    if MODEL_PATH.exists():
        return MODEL_PATH
    return sym_path


# ==============================================================================
# Helper Utilities
# ==============================================================================
def is_crypto_symbol(symbol: str = SYMBOL) -> bool:
    """Detect whether symbol represents a cryptocurrency pair."""
    s = symbol.upper()
    return "/" in s or "-" in s or (s.endswith("USD") and any(coin in s for coin in ["BTC", "ETH", "SOL", "LTC", "AVAX", "DOGE"]))


def setup_logger(name: str = "TradingAgent") -> logging.Logger:
    """
    Configures and returns a thread-safe logger writing to both stdout and trading_agent.log.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
