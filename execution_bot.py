"""
Execution Bot Module for Autonomous Quantitative AI Trading Agent.
Integrates LocalPaperBroker for 100% standalone, Indonesia-ready paper trading execution
with real live market data via Yahoo Finance, fee modeling, and dashboard auto-generation.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from config import (
    ACTIVE_CRYPTO_POSITIONS_FILE,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    CONFIDENCE_THRESHOLD,
    DASHBOARD_FILE,
    DEFAULT_SLIPPAGE_BPS,
    HORIZON_BARS,
    LOCAL_ACCOUNT_FILE,
    LOOKBACK_DAYS,
    MAX_DAILY_DRAWDOWN,
    MAX_POSITION_EXPOSURE,
    PAPER_TRADING,
    RETRAIN_INTERVAL_DAYS,
    SYMBOL,
    SYMBOLS,
    TELEMETRY_FILE,
    TRADE_HISTORY_FILE,
    has_valid_alpaca_credentials,
    is_crypto_symbol,
    setup_logger,
)
from data_pipeline import engineer_features, fetch_historical_bars, get_latest_price
from local_paper_broker import LocalPaperBroker
from model_engine import QuantModel, create_triple_barrier_labels
from risk_allocator import calculate_optimal_allocation

logger = setup_logger("ExecutionBot")


class AlpacaExecutionBot:
    """
    Automated execution engine with Dual-Mode and Multi-Asset Support:
    1. Standalone Local Paper Broker (Default): 100% offline-ready, no API keys needed,
       persistent ledger, trailing breakeven, and auto-generated HTML dashboard.
    2. Multi-Asset Scanning: Scans and trades BTC-USD, ETH-USD, and SOL-USD concurrently.
    3. Alpaca Paper Trading (Optional): Interfaces with Alpaca SDK when valid keys are provided.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = PAPER_TRADING,
        symbol: str = SYMBOL,
        symbols: Optional[List[str]] = None,
        positions_file: Path = ACTIVE_CRYPTO_POSITIONS_FILE,
        account_file: Path = LOCAL_ACCOUNT_FILE,
        history_file: Path = TRADE_HISTORY_FILE,
        dashboard_file: Path = DASHBOARD_FILE,
        telemetry_file: Path = TELEMETRY_FILE,
        fee_bps: float = DEFAULT_SLIPPAGE_BPS,
    ):
        if symbols:
            self.symbols = list(symbols)
            self.symbol = self.symbols[0]
        else:
            self.symbol = symbol
            self.symbols = [symbol]
        self.is_crypto = is_crypto_symbol(self.symbol)

        # Initialize Standalone Local Paper Broker
        self.local_broker = LocalPaperBroker(
            account_file=account_file,
            history_file=history_file,
            positions_file=positions_file,
            dashboard_file=dashboard_file,
            telemetry_file=telemetry_file,
            fee_bps=fee_bps,
        )

        # Optional Alpaca Client
        self.api_key = api_key if api_key else (ALPACA_API_KEY if has_valid_alpaca_credentials() else "")
        self.secret_key = secret_key if secret_key else (ALPACA_SECRET_KEY if has_valid_alpaca_credentials() else "")
        self.paper = paper
        self.trading_client = None
        self.last_retrain_time = time.time()
        self._init_alpaca_client()

    def _init_alpaca_client(self) -> None:
        """Attempts optional Alpaca connection if credentials are configured."""
        if not self.api_key or not self.secret_key or not has_valid_alpaca_credentials():
            logger.info(
                "Running in STANDALONE PAPER BROKER MODE (100% self-contained, live yfinance pricing, no Alpaca keys needed)."
            )
            return

        try:
            from alpaca.trading.client import TradingClient
            self.trading_client = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=self.paper)
            account = self.trading_client.get_account()
            logger.info("Connected to Alpaca Paper Trading Account (Equity: $%s)", f"{float(account.equity):,.2f}")
        except Exception as exc:
            logger.warning("Could not connect to Alpaca (%s). Falling back to Local Paper Broker.", exc)
            self.trading_client = None

    def check_circuit_breaker(self) -> bool:
        """Checks daily intra-day drawdown (-3% threshold)."""
        return self.local_broker.check_circuit_breaker()

    def get_open_position(self, symbol: Optional[str] = None) -> Optional[Any]:
        """Queries active position for symbol."""
        target_symbol = symbol or self.symbol
        positions = self.local_broker._load_positions()
        if target_symbol in positions:
            return positions[target_symbol]
        return None

    def manage_open_crypto_positions(self, current_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """Delegates position management and trailing breakeven to LocalPaperBroker."""
        return self.local_broker.manage_open_positions(current_price=current_price)

    def execute_crypto_order(
        self,
        qty: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        atr: float,
        symbol: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Executes crypto buy order via local broker and optional Alpaca client."""
        target_symbol = symbol or self.symbol
        logger.info(
            "Submitting Crypto-Native Order: BUY %.4f %s @ $%.2f | SL=$%.2f | TP=$%.2f",
            qty, target_symbol, entry_price, stop_loss_price, take_profit_price,
        )

        res = self.local_broker.execute_buy_order(
            symbol=target_symbol,
            qty=qty,
            price=entry_price,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            atr=atr,
        )

        # Mirror to Alpaca Paper if client is active
        if self.trading_client is not None and res is not None:
            try:
                from alpaca.trading.enums import OrderSide, TimeInForce
                from alpaca.trading.requests import MarketOrderRequest
                alpaca_order = MarketOrderRequest(
                    symbol=target_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )
                self.trading_client.submit_order(alpaca_order)
                logger.info("Mirrored order to Alpaca Paper Trading successfully.")
            except Exception as e:
                logger.warning("Alpaca order mirror skipped: %s", e)

        return res

    def run_cycle(
        self,
        model: Union[QuantModel, Dict[str, QuantModel]],
        sentiment_filter: Optional[Any] = None,
        use_synthetic: bool = False,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes one full autonomous cycle across target crypto symbols:
        1. Manage Existing Open Positions (SL, TP, Time, Trailing Breakeven)
        2. Evaluate Circuit Breaker (-3% Drawdown)
        3. Scan Target Assets: Pull bars, compute features, infer signals, evaluate sentiment, allocate capital, execute orders.
        4. Auto-Generate Dashboard HTML
        """
        target_symbols = symbols or self.symbols
        logger.info("=== Beginning Execution Cycle for Assets: %s ===", target_symbols)

        # 1. Manage Active Positions across all assets
        closed_events = self.manage_open_crypto_positions()

        # 2. Circuit Breaker check
        if self.check_circuit_breaker():
            self.local_broker.generate_dashboard_html()
            return {"status": "CIRCUIT_BREAKER_TRIGGERED", "closed_events": closed_events}

        # 3. Model Age Check: Trigger autonomous self-learning retraining if model > 7 days old
        if isinstance(model, dict):
            due_symbols = [
                s for s in target_symbols
                if s in model and hasattr(model[s], "is_retrain_due") and model[s].is_retrain_due()
            ]
            if due_symbols:
                logger.info(
                    "=== AUTO-RETRAIN TRIGGER: Models for %s exceed %d days age. Executing Self-Learning Retraining ===",
                    due_symbols, RETRAIN_INTERVAL_DAYS,
                )
                self.run_self_learning_retrain(
                    models=model,
                    symbols=due_symbols,
                    lookback_days=LOOKBACK_DAYS,
                    threshold=CONFIDENCE_THRESHOLD,
                    use_synthetic=use_synthetic,
                )

        scan_results: Dict[str, Any] = {}
        telemetry_assets: Dict[str, Any] = {}

        for sym in target_symbols:
            # Check existing position for this symbol
            existing_pos = self.get_open_position(sym)
            if existing_pos:
                live_p = get_latest_price(sym) or existing_pos["entry_price"]
                pnl = (live_p - existing_pos["entry_price"]) * existing_pos["qty"]
                logger.info(
                    "Active position present: %.4f %s @ $%.2f (Live: $%.2f | PnL: $%+.2f). Holding.",
                    existing_pos["qty"], sym, existing_pos["entry_price"], live_p, pnl,
                )
                scan_results[sym] = {"status": "IN_POSITION", "position": existing_pos}
                telemetry_assets[sym] = {
                    "price": round(live_p, 2),
                    "win_prob": 0.50,
                    "threshold": CONFIDENCE_THRESHOLD,
                    "signal": "HOLD",
                    "sentiment_score": 0.0,
                    "sentiment_label": "Netral",
                }
                continue

            # Fetch latest bars
            raw_df = fetch_historical_bars(symbol=sym, lookback_days=15, use_synthetic=use_synthetic)
            feature_df = engineer_features(raw_df)

            if feature_df.empty:
                logger.warning("Feature engineering produced empty dataset for %s.", sym)
                scan_results[sym] = {"status": "NO_DATA"}
                continue

            # Model Inference
            sym_model = model[sym] if isinstance(model, dict) and sym in model else (
                model if isinstance(model, QuantModel) else list(model.values())[0]
            )
            signal, win_prob = sym_model.predict_signal(feature_df)
            last_close = float(feature_df["close"].iloc[-1])
            last_atr = float(feature_df["atr"].iloc[-1])

            live_price = get_latest_price(sym)
            exec_price = live_price if (live_price and live_price > 0) else last_close

            # Sentiment Filter Gatekeeper
            effective_win_prob = win_prob
            sentiment_info = {}
            if sentiment_filter is not None:
                is_approved, adj_prob, sent_score, reason = sentiment_filter.evaluate_sentiment_gate(
                    symbol=sym, base_win_prob=win_prob
                )
                sentiment_info = {
                    "score": sent_score,
                    "approved": is_approved,
                    "reason": reason,
                    "original_win_prob": win_prob,
                    "effective_win_prob": adj_prob,
                }
                effective_win_prob = adj_prob
            else:
                sent_score = 0.0
                is_approved = True

            sent_label = "Positif" if sent_score > 0.15 else ("Negatif" if sent_score < -0.15 else "Netral")
            telemetry_assets[sym] = {
                "price": round(exec_price, 2),
                "win_prob": round(win_prob, 4),
                "threshold": CONFIDENCE_THRESHOLD,
                "signal": signal,
                "sentiment_score": round(sent_score, 2),
                "sentiment_label": sent_label,
            }

            if signal != "BUY":
                logger.info("Model Signal for %s: %s (Win Prob: %.2f%%). Cash preserved.", sym, signal, win_prob * 100.0)
                scan_results[sym] = {"status": "HOLD", "win_prob": win_prob, "sentiment": sentiment_info}
                continue

            if not is_approved:
                logger.warning("[%s] Signal rejected by Sentiment Gatekeeper: %s", sym, reason)
                scan_results[sym] = {"status": "SENTIMENT_VETOED", "sentiment": sentiment_info}
                continue

            # Half-Kelly Risk Allocation (50% max position exposure per asset)
            summary = self.local_broker.get_account_summary()
            alloc = calculate_optimal_allocation(
                equity=summary["equity"],
                buying_power=summary["cash"],
                entry_price=exec_price,
                atr=last_atr,
                win_prob=effective_win_prob,
                symbol=sym,
                max_exposure_pct=MAX_POSITION_EXPOSURE,
            )

            qty = alloc.get("qty", 0.0)
            if qty <= 0 or alloc.get("status") != "APPROVED":
                logger.info("[%s] Allocation rejected or qty is 0. Status: %s", sym, alloc.get("status"))
                scan_results[sym] = {"status": "ALLOCATION_REJECTED", "allocation": alloc}
                continue

            # Order Execution
            order = self.execute_crypto_order(
                qty=qty,
                entry_price=exec_price,
                stop_loss_price=alloc["stop_loss_price"],
                take_profit_price=alloc["take_profit_price"],
                atr=last_atr,
                symbol=sym,
            )

            scan_results[sym] = {
                "status": "ORDER_SUBMITTED",
                "order": order,
                "allocation": alloc,
                "sentiment": sentiment_info,
            }

        telemetry_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold": CONFIDENCE_THRESHOLD,
            "assets": telemetry_assets,
        }
        self.local_broker.generate_dashboard_html(telemetry=telemetry_payload)
        return {
            "status": "COMPLETED",
            "symbols": scan_results,
            "closed_events": closed_events,
            "telemetry": telemetry_payload,
        }

    def run_self_learning_retrain(
        self,
        models: Dict[str, QuantModel],
        symbols: Optional[List[str]] = None,
        lookback_days: int = LOOKBACK_DAYS,
        threshold: float = CONFIDENCE_THRESHOLD,
        use_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes autonomous Self-Learning Retraining across target crypto models:
        1. Pulls historical market data.
        2. Applies Failure Experience Replay weighting based on trade_history.csv.
        3. Trains new XGBoost model and checks Out-of-Sample Quality Gatekeeper.
        4. Reloads accepted models in-memory without interrupting live execution loop.
        5. Refreshes models/dashboard.html with updated retrain audit info.
        """
        target_symbols = symbols or self.symbols
        logger.info("=== STARTING AUTONOMOUS SELF-LEARNING RETRAINING FOR: %s ===", target_symbols)
        retrain_results: Dict[str, Any] = {}

        for sym in target_symbols:
            logger.info("Retraining self-learning model for %s (lookback: %d days)...", sym, lookback_days)
            try:
                raw_df = fetch_historical_bars(symbol=sym, lookback_days=lookback_days, use_synthetic=use_synthetic)
                feature_df = engineer_features(raw_df)
                labeled_df = create_triple_barrier_labels(feature_df)

                sym_model = models.get(sym)
                if sym_model is None:
                    sym_model = QuantModel(confidence_threshold=threshold, symbol=sym)
                    models[sym] = sym_model

                metrics = sym_model.train_and_evaluate(labeled_df, train_ratio=0.80, symbol=sym)
                # In-memory reload of the active model
                sym_model.load_model()
                retrain_results[sym] = {
                    "status": metrics.get("retrain_status", "ACCEPTED"),
                    "gatekeeper_reason": metrics.get("gatekeeper_reason", ""),
                    "metrics": metrics,
                }
                logger.info(
                    "[%s] Self-Learning Retrain Complete: Status=%s | Replayed Trades=%d (Penalties: %d, Boosts: %d)",
                    sym, metrics.get("retrain_status"), metrics.get("trades_matched", 0),
                    metrics.get("penalties_count", 0), metrics.get("boosts_count", 0),
                )
            except Exception as exc:
                logger.error("Self-learning retrain failed for %s: %s", sym, exc)
                retrain_results[sym] = {"status": "ERROR", "error": str(exc)}

        self.last_retrain_time = time.time()
        self.local_broker.generate_dashboard_html()
        return retrain_results
