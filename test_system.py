"""
Comprehensive Integration and Unit Test Suite for Autonomous Quantitative AI Trading Agent.
Validates:
1. Configuration and Risk Constants
2. Data Pipeline & Feature Stationarity (No Data Leakage / No Raw Prices)
3. Triple-Barrier Labeling & QuantModel Training/Inference
4. Fractional Kelly & ATR Volatility Risk Allocator
5. Out-of-Sample Backtester with 15 bps Crypto Slippage
6. Circuit Breaker & Execution Safety Guardrails
7. Crypto-Native Client-Side Bracket & Time-Barrier Manager (SL, TP, Time, Trailing Breakeven)
8. NLP Sentiment Filter Gatekeeper (FinBERT & Lexicon Veto/Boost)
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from config import (
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    CONFIDENCE_THRESHOLD,
    DEFAULT_SLIPPAGE_BPS,
    FAILURE_SAMPLE_WEIGHT,
    HORIZON_BARS,
    MAX_DAILY_DRAWDOWN,
    MAX_POSITION_EXPOSURE,
    MAX_RISK_PER_TRADE,
    MIN_RISK_PER_TRADE,
    RETRAIN_HISTORY_FILE,
    SENTIMENT_BOOST_THRESHOLD,
    SENTIMENT_BOOST_VALUE,
    SENTIMENT_VETO_THRESHOLD,
    SUCCESS_SAMPLE_WEIGHT,
    is_crypto_symbol,
)
from data_pipeline import (
    STATIONARY_FEATURE_COLUMNS,
    engineer_features,
    generate_synthetic_market_data,
)
from model_engine import (
    QuantModel,
    apply_experience_replay,
    create_triple_barrier_labels,
    extract_trade_feedback,
    load_retrain_history,
)
from risk_allocator import calculate_optimal_allocation
from backtester import FastBacktester
from execution_bot import AlpacaExecutionBot
from sentiment_filter import CryptoSentimentFilter


class TestQuantitativeTradingAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate controlled deterministic market data for testing
        cls.raw_df = generate_synthetic_market_data(symbol="BTC/USD", days=90, freq="1h")
        cls.feature_df = engineer_features(cls.raw_df)
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_config_guardrails(self):
        """Verify strict risk parameters and asset detection."""
        self.assertTrue(is_crypto_symbol("BTC/USD"))
        self.assertTrue(is_crypto_symbol("ETH/USD"))
        self.assertFalse(is_crypto_symbol("SPY"))
        self.assertFalse(is_crypto_symbol("AAPL"))

        # Reward to risk ratio must be 2:1
        ratio = ATR_TP_MULTIPLIER / ATR_SL_MULTIPLIER
        self.assertAlmostEqual(ratio, 2.0, places=2)

        # Risk parameters sanity
        self.assertEqual(MAX_RISK_PER_TRADE, 0.01)
        self.assertEqual(MAX_POSITION_EXPOSURE, 0.50)
        self.assertEqual(MAX_DAILY_DRAWDOWN, 0.03)
        self.assertEqual(HORIZON_BARS, 24)
        self.assertEqual(CONFIDENCE_THRESHOLD, 0.54)

        # Crypto fee default must be 15 bps (0.0015)
        self.assertEqual(DEFAULT_SLIPPAGE_BPS, 0.0015)

    def test_02_feature_engineering_stationarity(self):
        """Verify features are strictly stationary, have no NaNs, and do NOT use raw close prices."""
        # 1. Check feature columns
        for col in STATIONARY_FEATURE_COLUMNS:
            self.assertIn(col, self.feature_df.columns)

        # 2. Raw close must NOT be in the model's feature set
        self.assertNotIn("close", STATIONARY_FEATURE_COLUMNS)
        self.assertNotIn("open", STATIONARY_FEATURE_COLUMNS)
        self.assertNotIn("high", STATIONARY_FEATURE_COLUMNS)
        self.assertNotIn("low", STATIONARY_FEATURE_COLUMNS)

        # 3. Clean dataframe has no nulls
        feature_matrix = self.feature_df[STATIONARY_FEATURE_COLUMNS]
        self.assertEqual(feature_matrix.isna().sum().sum(), 0)

        # 4. Normalized features bound checks
        self.assertTrue((self.feature_df["rsi_14_norm"] >= 0.0).all())
        self.assertTrue((self.feature_df["rsi_14_norm"] <= 1.0).all())
        self.assertTrue((self.feature_df["atr_norm"] > 0.0).all())

    def test_03_triple_barrier_labeling(self):
        """Verify Triple-Barrier labeling returns binary targets without lookahead leak."""
        labeled_df = create_triple_barrier_labels(
            self.feature_df,
            atr_sl_mult=ATR_SL_MULTIPLIER,
            atr_tp_mult=ATR_TP_MULTIPLIER,
            horizon=HORIZON_BARS,
        )

        # Labels must be binary {0, 1}
        unique_labels = set(labeled_df["target"].unique())
        self.assertTrue(unique_labels.issubset({0, 1}))

        # Length must be truncated by horizon bars to avoid incomplete forward horizon bias
        self.assertEqual(len(labeled_df), len(self.feature_df) - HORIZON_BARS)

    def test_04_fractional_kelly_risk_allocator(self):
        """Test Quarter-Kelly allocation, ATR stop/profit prices, and exposure limits."""
        equity = 100_000.0
        buying_power = 100_000.0
        entry_price = 50_000.0
        atr = 500.0

        # Scenario A: High win probability (e.g. 70%) -> Positive Kelly fraction
        win_prob = 0.70
        alloc = calculate_optimal_allocation(
            equity=equity,
            buying_power=buying_power,
            entry_price=entry_price,
            atr=atr,
            win_prob=win_prob,
        )

        self.assertEqual(alloc["status"], "APPROVED")
        # SL = entry - 1.5 * 500 = 50,000 - 750 = 49,250
        self.assertEqual(alloc["stop_loss_price"], 49250.0)
        # TP = entry + 3.0 * 500 = 50,000 + 1500 = 51,500
        self.assertEqual(alloc["take_profit_price"], 51500.0)

        # Risk must be capped at MAX_RISK_PER_TRADE (1% = $1,000)
        self.assertLessEqual(alloc["risk_pct"], MAX_RISK_PER_TRADE)
        self.assertGreaterEqual(alloc["risk_pct"], MIN_RISK_PER_TRADE)
        self.assertLessEqual(alloc["risk_amount"], equity * MAX_RISK_PER_TRADE + 1e-4)

        # Position value must not exceed MAX_POSITION_EXPOSURE (25% = $25,000)
        self.assertLessEqual(alloc["position_value"], equity * MAX_POSITION_EXPOSURE + 1e-4)

        # Scenario B: Low win probability (e.g. 30%) -> Negative Kelly -> Reject trade
        low_win_prob = 0.30
        alloc_low = calculate_optimal_allocation(
            equity=equity,
            buying_power=buying_power,
            entry_price=entry_price,
            atr=atr,
            win_prob=low_win_prob,
        )
        self.assertEqual(alloc_low["status"], "NO_EDGE")
        self.assertEqual(alloc_low["qty"], 0.0)

    def test_05_model_engine_training_and_inference(self):
        """Test chronological model training, artifact persistence, and inference."""
        labeled_df = create_triple_barrier_labels(self.feature_df)
        model = QuantModel(confidence_threshold=CONFIDENCE_THRESHOLD)

        # Train with chronological split
        metrics = model.train_and_evaluate(labeled_df, train_ratio=0.80)
        self.assertIn("test_roc_auc", metrics)
        self.assertIn("confident_win_rate_pct", metrics)

        # Test inference
        signal, win_prob = model.predict_signal(self.feature_df)
        self.assertIn(signal, ["BUY", "HOLD"])
        self.assertTrue(0.0 <= win_prob <= 1.0)
        if win_prob >= CONFIDENCE_THRESHOLD:
            self.assertEqual(signal, "BUY")
        else:
            self.assertEqual(signal, "HOLD")

    def test_06_backtester_simulation(self):
        """Test backtester execution on out-of-sample data with 15 bps slippage."""
        labeled_df = create_triple_barrier_labels(self.feature_df)
        model = QuantModel(confidence_threshold=CONFIDENCE_THRESHOLD)
        model.train_and_evaluate(labeled_df, train_ratio=0.80)

        split_idx = int(len(self.feature_df) * 0.80)
        test_df = self.feature_df.iloc[split_idx:].copy()

        backtester = FastBacktester(initial_capital=50_000.0, symbol="BTC/USD", slippage_bps=0.0015)
        results = backtester.run_backtest(test_df=test_df, model=model)

        self.assertIn("total_return_pct", results)
        self.assertIn("win_rate_pct", results)
        self.assertIn("profit_factor", results)
        self.assertIn("sharpe_ratio", results)
        self.assertIn("max_drawdown_pct", results)
        self.assertEqual(results["initial_capital"], 50_000.0)

    def test_07_sentiment_filter_veto_and_boost(self):
        """Test CryptoSentimentFilter: Veto on FUD, Boost on strong positive news."""
        sent_filter = CryptoSentimentFilter()

        # 1. Strongly Bearish / FUD news sample
        bearish_news = [
            "SEC files massive lawsuit against major crypto exchanges for fraud and illegal activities.",
            "Bitcoin plunges and crashes amid panic selling, severe liquidation risk, and security hack.",
            "Crypto market bloodbath continues as regulatory ban threats increase outflows.",
        ]
        is_approved, adj_prob, score, reason = sent_filter.evaluate_sentiment_gate(
            symbol="BTC/USD", base_win_prob=0.65, sample_news=bearish_news
        )
        self.assertFalse(is_approved)
        self.assertLess(score, SENTIMENT_VETO_THRESHOLD)
        self.assertIn("VETOED", reason)

        # 2. Strongly Bullish news sample
        bullish_news = [
            "Bitcoin surges past key resistance with massive institutional adoption and record ETF inflows.",
            "Crypto rally gains momentum as bulls target new all-time high with breakthrough profit growth.",
            "Major financial institutions partner for global crypto accumulation surge.",
        ]
        is_approved_b, adj_prob_b, score_b, reason_b = sent_filter.evaluate_sentiment_gate(
            symbol="BTC/USD", base_win_prob=0.65, sample_news=bullish_news
        )
        self.assertTrue(is_approved_b)
        self.assertGreater(score_b, SENTIMENT_BOOST_THRESHOLD)
        self.assertAlmostEqual(adj_prob_b, 0.65 + SENTIMENT_BOOST_VALUE, places=2)
        self.assertIn("BOOSTED", reason_b)

        # 3. Neutral news sample
        neutral_news = [
            "Federal Reserve meeting concludes with interest rate policy announcement.",
            "Cryptocurrency trading volume stabilizes during weekend market session.",
        ]
        is_approved_n, adj_prob_n, score_n, reason_n = sent_filter.evaluate_sentiment_gate(
            symbol="BTC/USD", base_win_prob=0.65, sample_news=neutral_news
        )
        self.assertTrue(is_approved_n)
        self.assertAlmostEqual(adj_prob_n, 0.65, places=2)
        self.assertIn("APPROVED", reason_n)

    def test_08_crypto_client_side_bracket_manager(self):
        """Test Client-Side Bracket Manager: Trailing Breakeven, Stop Loss, Take Profit, Time Barrier."""
        test_pos_file = Path(self.temp_dir) / "test_active_positions.json"
        test_acc_file = Path(self.temp_dir) / "test_account.json"
        test_hist_file = Path(self.temp_dir) / "test_history.csv"
        test_dash_file = Path(self.temp_dir) / "test_dashboard.html"

        bot = AlpacaExecutionBot(
            symbol="BTC/USD",
            positions_file=test_pos_file,
            account_file=test_acc_file,
            history_file=test_hist_file,
            dashboard_file=test_dash_file,
        )

        # 1. Place initial crypto order
        entry_price = 50000.0
        atr = 1000.0
        # SL = 50000 - 1500 = 48500, TP = 50000 + 3000 = 53000
        bot.execute_crypto_order(
            qty=0.5,
            entry_price=entry_price,
            stop_loss_price=48500.0,
            take_profit_price=53000.0,
            atr=atr,
        )

        positions = bot.local_broker._load_positions()
        self.assertIn("BTC/USD", positions)
        self.assertEqual(positions["BTC/USD"]["stop_loss_price"], 48500.0)
        self.assertFalse(positions["BTC/USD"]["breakeven_triggered"])

        # 2. Test Trailing Breakeven: Price reaches entry + 1.5 * ATR = 51500 (halfway to TP)
        bot.manage_open_crypto_positions(current_price=51600.0)
        positions = bot.local_broker._load_positions()
        self.assertIn("BTC/USD", positions)
        # SL must now be moved up to Entry Price (50000.0)
        self.assertEqual(positions["BTC/USD"]["stop_loss_price"], 50000.0)
        self.assertTrue(positions["BTC/USD"]["breakeven_triggered"])

        # 3. Test Take Profit Hit: Price reaches 53500 (>= 53000)
        closed_tp = bot.manage_open_crypto_positions(current_price=53500.0)
        self.assertEqual(len(closed_tp), 1)
        self.assertEqual(closed_tp[0]["reason"], "TAKE_PROFIT")
        positions = bot.local_broker._load_positions()
        self.assertNotIn("BTC/USD", positions)

        # 4. Test Stop Loss Hit
        bot.execute_crypto_order(
            qty=0.5,
            entry_price=50000.0,
            stop_loss_price=48500.0,
            take_profit_price=53000.0,
            atr=atr,
        )
        closed_sl = bot.manage_open_crypto_positions(current_price=48000.0)
        self.assertEqual(len(closed_sl), 1)
        self.assertEqual(closed_sl[0]["reason"], "STOP_LOSS")

        # 5. Test Time Barrier Hit: Position held past HORIZON_BARS
        bot.execute_crypto_order(
            qty=0.5,
            entry_price=50000.0,
            stop_loss_price=48500.0,
            take_profit_price=53000.0,
            atr=atr,
        )
        # Manually backdate the entry_timestamp by 25 hours (horizon is 24 hours)
        positions = bot.local_broker._load_positions()
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        positions["BTC/USD"]["entry_timestamp"] = old_time.isoformat()
        bot.local_broker._save_positions(positions)

        closed_time = bot.manage_open_crypto_positions(current_price=50200.0)
        self.assertEqual(len(closed_time), 1)
        self.assertEqual(closed_time[0]["reason"], "TIME_BARRIER")
        positions = bot.local_broker._load_positions()
        self.assertNotIn("BTC/USD", positions)
        self.assertTrue(test_dash_file.exists())

    def test_09_failure_sample_weighting_and_experience_replay(self):
        """Test Failure Experience Replay weighting, target overriding, and Quality Gatekeeper."""
        labeled_df = create_triple_barrier_labels(self.feature_df)

        # 1. Select specific bars for simulated past trades
        loss_idx = 10
        win_idx = 25
        loss_dt = labeled_df.index[loss_idx]
        win_dt = labeled_df.index[win_idx]

        mock_trades = [
            {
                "symbol": "BTC/USD",
                "entry_timestamp": loss_dt.isoformat(),
                "exit_timestamp": (loss_dt + timedelta(hours=3)).isoformat(),
                "net_pnl": -250.0,
                "reason": "STOP_LOSS",
            },
            {
                "symbol": "BTC/USD",
                "entry_timestamp": win_dt.isoformat(),
                "exit_timestamp": (win_dt + timedelta(hours=5)).isoformat(),
                "net_pnl": +500.0,
                "reason": "TAKE_PROFIT",
            },
        ]

        # 2. Test apply_experience_replay
        sample_weights, df_mod, summary = apply_experience_replay(
            df=labeled_df,
            trades=mock_trades,
            failure_weight=FAILURE_SAMPLE_WEIGHT,
            success_weight=SUCCESS_SAMPLE_WEIGHT,
            window_hours=1.0,
        )

        self.assertEqual(summary["matched"], 2)
        self.assertGreaterEqual(summary["penalties"], 1)
        self.assertGreaterEqual(summary["boosts"], 1)

        # Loss bar must have weight 3.0 and target 0
        self.assertEqual(sample_weights[loss_idx], 3.0)
        self.assertEqual(df_mod["target"].iloc[loss_idx], 0)

        # Win bar must have weight 1.5 and target 1
        self.assertEqual(sample_weights[win_idx], 1.5)
        self.assertEqual(df_mod["target"].iloc[win_idx], 1)

        # Non-matched bars must retain baseline weight 1.0
        self.assertEqual(sample_weights[0], 1.0)
        self.assertEqual(sample_weights[50], 1.0)

        # 3. Test extract_trade_feedback from CSV pairing
        test_csv = Path(self.temp_dir) / "test_trades_replay.csv"
        with open(test_csv, "w", encoding="utf-8") as f:
            f.write("timestamp,symbol,action,price,qty,gross_val,fee,net_pnl,net_pnl_pct,reason\n")
            f.write(f"{loss_dt.strftime('%Y-%m-%d %H:%M:%S')},BTC-USD,BUY,50000.00,0.5000,25000.00,37.50,0.00,0.00%,SIGNAL_ENTRY\n")
            f.write(f"{(loss_dt + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')},BTC-USD,SELL - STOP_LOSS,48500.00,0.5000,24250.00,36.38,-787.50,-3.00%,STOP_LOSS\n")

        extracted = extract_trade_feedback(trade_history=test_csv, symbol="BTC-USD")
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["entry_timestamp"], loss_dt.strftime('%Y-%m-%d %H:%M:%S'))
        self.assertEqual(extracted[0]["net_pnl"], -787.50)

        # 4. Test model training with experience replay and Quality Gatekeeper
        test_model_path = Path(self.temp_dir) / "test_replay_model.joblib"
        model = QuantModel(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            model_path=str(test_model_path),
            symbol="BTC-USD",
        )
        metrics = model.train_and_evaluate(
            df=labeled_df,
            train_ratio=0.80,
            symbol="BTC-USD",
            trade_history=mock_trades,
        )

        self.assertIn("retrain_status", metrics)
        self.assertEqual(metrics["retrain_status"], "ACCEPTED")
        self.assertEqual(metrics["trades_matched"], 2)
        self.assertTrue(test_model_path.exists())

        # Verify audit logging in retrain_history.json
        history = load_retrain_history()
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)
        self.assertIn("experience_replay", history[-1])


if __name__ == "__main__":
    unittest.main()
