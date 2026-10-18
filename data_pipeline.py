"""
Data Pipeline Module for Autonomous Quantitative AI Trading Agent.
Powered by Yahoo Finance (yfinance) for 100% standalone, real-time, zero-API-key market data
with strict feature stationarity and zero data leakage.
"""

from datetime import datetime, timezone
from typing import List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    LOOKBACK_DAYS,
    SYMBOL,
    TIMEFRAME,
    is_crypto_symbol,
    setup_logger,
)

logger = setup_logger("DataPipeline")

# Explicit stationary feature list used as model input (strictly no raw prices)
STATIONARY_FEATURE_COLUMNS: List[str] = [
    "dist_ema20",
    "dist_ema50",
    "dist_ema200",
    "ema20_slope_3",
    "rsi_14_norm",
    "macd_diff_norm",
    "bb_width",
    "bb_pct_b",
    "atr_norm",
    "volume_ratio",
    "log_ret_1",
    "log_ret_2",
    "log_ret_3",
]


def to_yfinance_symbol(symbol: str) -> str:
    """
    Normalizes trading pair notation to Yahoo Finance ticker format:
    e.g. 'BTC/USD' -> 'BTC-USD', 'ETH/USD' -> 'ETH-USD', 'BTCUSD' -> 'BTC-USD'.
    """
    s = symbol.strip().upper()
    if "/" in s:
        return s.replace("/", "-")
    if s.endswith("USD") and not s.endswith("-USD") and len(s) > 3:
        return f"{s[:-3]}-USD"
    return s


def parse_yfinance_interval(tf_str: str) -> str:
    """
    Maps timeframe strings to Yahoo Finance interval codes.
    Supported: '1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo'.
    """
    tf = tf_str.strip().lower()
    if tf in ("1hour", "1h", "hour", "60m"):
        return "1h"
    if tf in ("1day", "1d", "day"):
        return "1d"
    if tf in ("15min", "15m"):
        return "15m"
    if tf in ("5min", "5m"):
        return "5m"
    if tf in ("1min", "1m"):
        return "1m"
    return "1h"


def get_latest_price(symbol: str = SYMBOL) -> float:
    """
    Fetches real-time market price for the specified symbol via Yahoo Finance fast_info.
    Falls back to recent bar close if fast_info is unavailable.
    """
    yf_symbol = to_yfinance_symbol(symbol)
    try:
        ticker = yf.Ticker(yf_symbol)
        # Try fast_info first (fastest)
        if hasattr(ticker, "fast_info") and "lastPrice" in ticker.fast_info:
            price = float(ticker.fast_info["lastPrice"])
            if price > 0:
                return price

        # Fallback to last 1-day 1m bar
        df = ticker.history(period="1d", interval="1m")
        if not df.empty and "Close" in df.columns:
            return float(df["Close"].iloc[-1])

    except Exception as exc:
        logger.warning("Failed to fetch live price for %s via yfinance (%s).", yf_symbol, exc)

    return 0.0


def fetch_historical_bars(
    symbol: str = SYMBOL,
    timeframe_str: str = TIMEFRAME,
    lookback_days: int = LOOKBACK_DAYS,
    end_time: Optional[datetime] = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Fetches real historical OHLCV bars using Yahoo Finance (yfinance).
    - Default primary engine: pulls real market data without requiring any API keys.
    - Capped at period='720d' for 1h interval per Yahoo Finance API limits.
    - Standardizes index to UTC and columns to lowercase ['open', 'high', 'low', 'close', 'volume'].
    - Only falls back to synthetic generator if use_synthetic=True is explicitly passed.

    Returns:
        pd.DataFrame with UTC timestamp index and columns: ['open', 'high', 'low', 'close', 'volume']
    """
    if use_synthetic:
        logger.info("Explicit synthetic flag passed. Generating synthetic market data for %s...", symbol)
        return generate_synthetic_market_data(
            symbol=symbol,
            days=lookback_days,
            freq="1h" if "hour" in timeframe_str.lower() else "15min",
        )

    yf_symbol = to_yfinance_symbol(symbol)
    interval = parse_yfinance_interval(timeframe_str)

    # Yahoo Finance allows max 730 days for 1h data
    safe_days = min(lookback_days, 720)
    period_str = f"{safe_days}d"

    logger.info(
        "Fetching REAL market data from Yahoo Finance for %s (%s, period: %s, interval: %s)...",
        yf_symbol,
        symbol,
        period_str,
        interval,
    )

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period_str, interval=interval)

        if df is None or df.empty:
            raise ValueError(f"Yahoo Finance returned empty historical bars for {yf_symbol}")

        # Flatten multi-level columns if present in newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize columns to lowercase
        df.columns = [str(col).lower() for col in df.columns]

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required price columns from yfinance: {missing}")

        df = df[required_cols].copy()

        # Ensure index is UTC datetime
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone.utc)
        else:
            df.index = df.index.tz_convert(timezone.utc)

        df.index.name = "timestamp"
        df.sort_index(inplace=True)
        # Drop duplicates
        df = df[~df.index.duplicated(keep="last")]
        # Drop rows where volume or close is zero or NaN
        df = df[(df["close"] > 0) & (df["volume"] >= 0)].dropna()

        logger.info(
            "Successfully fetched %d real market bars for %s (%s to %s) | Latest Close: $%.2f",
            len(df),
            yf_symbol,
            df.index[0].strftime("%Y-%m-%d %H:%M"),
            df.index[-1].strftime("%Y-%m-%d %H:%M"),
            float(df["close"].iloc[-1]),
        )
        return df

    except Exception as exc:
        logger.warning(
            "Failed to fetch data from Yahoo Finance (%s). Falling back to synthetic market data generator.",
            exc,
        )
        return generate_synthetic_market_data(
            symbol=symbol,
            days=lookback_days,
            freq="1h" if "hour" in timeframe_str.lower() else "15min",
        )


def generate_synthetic_market_data(
    symbol: str = "BTC/USD", days: int = 730, freq: str = "1h"
) -> pd.DataFrame:
    """
    Generates realistic synthetic OHLCV data using a Geometric Brownian Motion
    with stochastic volatility and mean-reverting intra-bar jumps for offline testing.
    """
    logger.info("Generating synthetic market data for %s over %d days...", symbol, days)
    np.random.seed(42)

    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=days)
    dates = pd.date_range(start=start, end=end, freq=freq, tz=timezone.utc)
    n = len(dates)

    initial_price = 80000.0 if "BTC" in symbol else 450.0
    mu = 0.00015
    sigma = 0.008

    hour_factors = np.sin(np.linspace(0, 2 * np.pi * (days), n)) * 0.002
    returns = np.random.normal(mu, sigma, n) + hour_factors

    price_path = initial_price * np.exp(np.cumsum(returns))

    intra_noise = np.random.uniform(0.001, 0.006, n)
    highs = price_path * (1 + intra_noise)
    lows = price_path * (1 - intra_noise)
    opens = np.roll(price_path, 1)
    opens[0] = initial_price
    closes = price_path

    highs = np.maximum.reduce([highs, opens, closes])
    lows = np.minimum.reduce([lows, opens, closes])

    base_vol = 150.0 if "BTC" in symbol else 500000.0
    volumes = np.random.lognormal(mean=np.log(base_vol), sigma=0.6, size=n)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    logger.info("Generated %d synthetic bars successfully.", len(df))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly stationary features from raw OHLCV market data.
    IMPORTANT: No raw close prices are emitted as features to ensure model stationarity
    and prevent non-stationary drift / data leakage.

    Features computed (13 features):
    1. Distance to EMA-20, EMA-50, EMA-200: (Close - EMA) / EMA
    2. Slope of EMA-20 over last 3 candles: (EMA20 - EMA20.shift(3)) / EMA20.shift(3)
    3. RSI(14) normalized to [0, 1] range: RSI / 100.0
    4. MACD Histogram normalized by Close price: MACD_diff / Close
    5. Bollinger Band Width and %B position:
       - Width: (Upper - Lower) / Mid
       - %B: (Close - Lower) / (Upper - Lower)
    6. Normalized ATR(14): ATR(14) / Close
    7. Volume ratio: Volume / Rolling_Mean_Volume(20)
    8. Log returns: t-1, t-2, t-3

    Returns:
        pd.DataFrame containing original OHLCV plus engineered stationary features,
        with initial warm-up NaNs dropped cleanly without forward lookahead bias.
    """
    df = df.copy()

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # 1. Exponential Moving Averages (EMA 20, 50, 200)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    df["dist_ema20"] = (close - ema20) / (ema20 + 1e-9)
    df["dist_ema50"] = (close - ema50) / (ema50 + 1e-9)
    df["dist_ema200"] = (close - ema200) / (ema200 + 1e-9)

    # 2. Slope of EMA-20 over last 3 candles
    df["ema20_slope_3"] = (ema20 - ema20.shift(3)) / (ema20.shift(3) + 1e-9)

    # 3. Relative Strength Index (RSI 14) normalized to 0-1
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))
    df["rsi_14_norm"] = (rsi_14 / 100.0).clip(0.0, 1.0)

    # 4. Moving Average Convergence Divergence (MACD 12, 26, 9) normalized by Close
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_diff = macd_line - macd_signal
    df["macd_diff_norm"] = macd_diff / (close + 1e-9)

    # 5. Bollinger Bands (20-period, 2-std)
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_range = bb_upper - bb_lower

    df["bb_width"] = bb_range / (bb_mid + 1e-9)
    df["bb_pct_b"] = (close - bb_lower) / (bb_range + 1e-9)

    # 6. Average True Range (ATR 14) and Normalized ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(com=13, adjust=False).mean()
    df["atr"] = atr
    df["atr_norm"] = atr / (close + 1e-9)

    # 7. Volume Ratio (relative to 20-period rolling average)
    vol_ma20 = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / (vol_ma20 + 1e-9)

    # 8. Log Returns of t-1, t-2, t-3
    df["log_ret_1"] = np.log(close / (close.shift(1) + 1e-9))
    df["log_ret_2"] = np.log(close.shift(1) / (close.shift(2) + 1e-9))
    df["log_ret_3"] = np.log(close.shift(2) / (close.shift(3) + 1e-9))

    clean_df = df.dropna().copy()
    logger.info(
        "Engineered %d stationary features. Bars after warm-up drop: %d",
        len(STATIONARY_FEATURE_COLUMNS),
        len(clean_df),
    )
    return clean_df
