# Autonomous Quantitative AI Trading Agent (Crypto-Native & FinBERT)

A production-grade, autonomous quantitative machine learning trading system built in Python for **Crypto (BTC/USD, ETH/USD) and Equities on Alpaca Paper Trading**.

Features a self-trained **XGBoost Classifier** with Marcos Lopez de Prado's **Triple-Barrier Labeling**, dynamic capital allocation using **Fractional Kelly Criterion (Quarter-Kelly) + ATR Volatility Sizing**, an NLP Sentiment Gatekeeper powered by **Lazy FinBERT (`ProsusAI/finbert`)**, and a **Client-Side Crypto Bracket & Time-Barrier Manager** with **Trailing Breakeven**.

---

## 🌟 Key Architecture & Upgrades

### 1. Crypto-Native Client-Side Bracket & Time-Barrier Manager
Alpaca API returns error 422 if `OrderClass.BRACKET` or `TimeInForce.DAY` is sent for cryptocurrency pairs (`BTC/USD`, `ETH/USD`). To solve this cleanly:
- **Order Placement**: Crypto entry orders are sent as `MarketOrderRequest(side=OrderSide.BUY, time_in_force=TimeInForce.GTC)`.
- **Local State Tracking**: Position metadata is recorded into `models/active_crypto_positions.json` with entry price, quantity, stop loss, take profit, ATR, timestamp, and horizon duration.
- **Client-Side Barrier Execution**: Evaluated at the start of every cycle:
  - **Stop Loss Hit**: Market SELL (GTC) if price $\le StopLoss$.
  - **Take Profit Hit**: Market SELL (GTC) if price $\ge TakeProfit$.
  - **Time Barrier Hit**: Market SELL (GTC) if holding time $\ge 12\text{ bars}$ (`HORIZON_BARS`).
  - **Trailing Breakeven**: If price rallies past $Entry + (1.5 \times ATR)$ (halfway to TP), Stop Loss shifts up to $EntryPrice$ for a completely risk-free run.

### 2. Real-Time News Sentiment Gatekeeper (FinBERT NLP)
- **News Extraction**: Queries Alpaca `NewsClient` (`alpaca.data.historical.news`) for the 10 latest articles in the last 24 hours.
- **Lazy Loading**: FinBERT (`ProsusAI/finbert` via HuggingFace `transformers`) is loaded into memory only when an XGBoost `BUY` signal is generated, saving CPU and RAM.
- **Lightweight Lexicon Fallback**: Seamless fallback to a domain-specific financial lexicon if offline or in minimal environments.
- **Veto & Boost Rules**:
  - Composite Sentiment $< -0.15$: **VETO (Cancels BUY signal)** due to market FUD.
  - Composite Sentiment $> +0.25$: **BOOST (Adds $+0.03$ to model `win_prob`)** for stronger Kelly sizing (still capped by `MAX_RISK_PER_TRADE`).

### 3. Real Market Data Priority & Crypto Fee Adjustment
- **Real Data Priority**: By default, `data_pipeline.py` queries `CryptoHistoricalDataClient()` (which Alpaca allows without keys for crypto) for live/historical bars.
- **Synthetic Flag (`--synthetic`)**: Use `--synthetic` to explicitly run offline simulations with realistic geometric Brownian motion.
- **Crypto Fee Modeling**: Backtest slippage/commission adjusted to `0.0015` ($0.15\%$ per side / 15 bps).

### 4. Mathematical Risk & ML Pipeline
- **Triple-Barrier Labeling**: Upper barrier $Close + 3.0 \times ATR$, lower barrier $Close - 1.5 \times ATR$ (2:1 Reward-to-Risk), time barrier 12 bars.
- **Stationary Features**: 13 stationary features (EMA distance ratios, EMA slope, normalized RSI, normalized MACD diff, Bollinger Band width & %B, normalized ATR, volume ratio, log returns). Zero raw prices as features.
- **Zero Data Leakage**: Strict chronological 80/20 train/test split.
- **Fractional Kelly Criterion**: Quarter-Kelly formula $f^* = 0.25 \times \frac{p \cdot b - (1 - p)}{b}$ clamped between $0.25\%$ and $1.0\%$ risk per trade.
- **Drawdown Circuit Breaker**: Halts execution if intra-day equity drawdown drops $\le -3\%$.

---

## 📁 Repository Structure

```
quant-trading-agent/
├── .env.example            # Environment variables template (API keys, strategy parameters)
├── requirements.txt        # Python dependency manifest (includes transformers)
├── config.py               # Central configuration, risk parameters & logger
├── data_pipeline.py        # Real Alpaca historical data & stationary feature engineering
├── model_engine.py         # Triple-Barrier labeling & QuantModel (XGBoost)
├── risk_allocator.py       # Fractional Kelly & ATR volatility sizing engine
├── sentiment_filter.py     # Lazy FinBERT NLP news sentiment filter & lexicon fallback
├── execution_bot.py        # Alpaca execution bot & Client-Side Crypto Bracket Manager
├── backtester.py           # Out-of-sample event-driven simulation (15 bps slippage)
├── main.py                 # Unified CLI entry point (--mode train | backtest | live)
├── test_system.py          # 8 unit & integration tests
├── trading_agent.log       # Centralized runtime log file
└── models/
    ├── xgb_trading_model.joblib      # Serialized trained model artifact
    └── active_crypto_positions.json  # Client-side active position state
```

---

## 🚀 Getting Started

### 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Alpaca Paper Trading

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set your Alpaca API credentials:
```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
PAPER_TRADING=True
SYMBOL=BTC/USD
```

---

## 💻 CLI Usage

### A. Train the Model (`--mode train`)

```bash
# Uses real Alpaca data by default
python3 main.py --mode train --lookback 365 --threshold 0.62

# Or offline synthetic mode:
python3 main.py --mode train --synthetic --lookback 90 --threshold 0.62
```

### B. Out-of-Sample Backtest (`--mode backtest`)

```bash
# Default threshold 0.62 and 15 bps crypto slippage
python3 main.py --mode backtest --threshold 0.62 --capital 100000

# Offline synthetic mode:
python3 main.py --mode backtest --synthetic --threshold 0.62 --capital 100000
```

### C. Live Paper Trading (`--mode live`)

```bash
# Single execution check (cron friendly)
python3 main.py --mode live --once

# Continuous real-time polling loop (every 60s)
python3 main.py --mode live --interval 60

# Offline simulation pulse:
python3 main.py --mode live --synthetic --once
```

---

## 🧪 Testing

Run the automated test suite verifying all 8 core subsystems:

```bash
python3 test_system.py
```
*(Tests cover risk parameters, feature stationarity, Triple-Barrier labeling, Fractional Kelly, Out-of-Sample Backtester, FinBERT/Lexicon sentiment veto & boost, and Client-Side Crypto Bracket Manager with Trailing Breakeven).*
