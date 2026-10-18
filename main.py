"""
Autonomous Quantitative AI Trading Agent - CLI Entry Point.
Orchestrates multi-asset model training, portfolio out-of-sample backtesting,
dual-speed live paper execution loops with real-time FinBERT sentiment filtering.
"""

import argparse
import shutil
import sys
import time
from typing import Dict, List, Optional

from backtester import FastBacktester
from config import (
    CONFIDENCE_THRESHOLD,
    LOOKBACK_DAYS,
    MODEL_PATH,
    RETRAIN_INTERVAL_DAYS,
    SYMBOL,
    SYMBOLS,
    get_model_path,
    setup_logger,
)
from data_pipeline import engineer_features, fetch_historical_bars, get_latest_price
from execution_bot import AlpacaExecutionBot
from model_engine import QuantModel, create_triple_barrier_labels
from sentiment_filter import CryptoSentimentFilter

logger = setup_logger("Main")


def run_training_mode(
    symbols: List[str], lookback_days: int, threshold: float, use_synthetic: bool = False
) -> Dict[str, QuantModel]:
    """
    Mode: TRAIN
    1. Pulls historical OHLCV data for target symbols.
    2. Calculates stationary technical features.
    3. Generates Triple-Barrier target labels.
    4. Trains regularized XGBoost with chronological 80/20 split.
    5. Evaluates out-of-sample performance and persists model artifacts.
    """
    logger.info("=== STARTING MULTI-ASSET QUANTITATIVE MODEL TRAINING PIPELINE ===")
    logger.info("Target Symbols : %s", symbols)
    logger.info("Lookback Days  : %d", lookback_days)
    logger.info("Confidence Min : %.2f", threshold)
    logger.info("Data Source    : %s", "Synthetic Generator" if use_synthetic else "Yahoo Finance Real Market Data")

    trained_models: Dict[str, QuantModel] = {}

    for sym in symbols:
        logger.info("-" * 65)
        logger.info(">>> Training Model for Asset: %s <<<", sym)
        raw_df = fetch_historical_bars(symbol=sym, lookback_days=lookback_days, use_synthetic=use_synthetic)
        feature_df = engineer_features(raw_df)
        labeled_df = create_triple_barrier_labels(feature_df)

        sym_model_path = get_model_path(sym, for_training=True)
        model = QuantModel(confidence_threshold=threshold, model_path=str(sym_model_path), symbol=sym)
        model.train_and_evaluate(labeled_df, train_ratio=0.80)
        trained_models[sym] = model

        # Ensure default model path exists or is updated
        if ("BTC" in sym.upper() or len(symbols) == 1) and sym_model_path.resolve() != MODEL_PATH.resolve():
            shutil.copy(sym_model_path, MODEL_PATH)

    logger.info("=== ALL MODEL TRAINING PIPELINES COMPLETED SUCCESSFULLY ===")
    return trained_models


def run_backtest_mode(
    symbols: List[str], lookback_days: int, threshold: float, capital: float, use_synthetic: bool = False
) -> None:
    """
    Mode: BACKTEST
    Runs high-fidelity event-driven backtesting:
    - Single asset: individual report with Half-Kelly sizing and 15 bps fee.
    - Multiple assets: synchronized multi-asset portfolio simulation sharing $100k capital.
    """
    logger.info("=== STARTING OUT-OF-SAMPLE BACKTEST SIMULATION ===")
    logger.info("Assets         : %s", symbols)
    logger.info("Data Source    : %s", "Synthetic Generator" if use_synthetic else "Yahoo Finance Real Market Data")

    test_dfs: Dict[str, Any] = {}
    models: Dict[str, QuantModel] = {}

    for sym in symbols:
        raw_df = fetch_historical_bars(symbol=sym, lookback_days=lookback_days, use_synthetic=use_synthetic)
        feature_df = engineer_features(raw_df)

        # Strictly reserve the last 20% for out-of-sample testing
        split_idx = int(len(feature_df) * 0.80)
        test_dfs[sym] = feature_df.iloc[split_idx:].copy()

        sym_model_path = get_model_path(sym)
        m = QuantModel(confidence_threshold=threshold, model_path=str(sym_model_path), symbol=sym)
        try:
            m.load_model()
        except Exception:
            logger.warning("Could not load model for %s. Training new model first...", sym)
            train_df = feature_df.iloc[:split_idx].copy()
            labeled_train = create_triple_barrier_labels(train_df)
            m.train_and_evaluate(labeled_train, train_ratio=0.80)
        models[sym] = m

    if len(symbols) == 1:
        sym = symbols[0]
        backtester = FastBacktester(initial_capital=capital, symbol=sym)
        backtester.run_backtest(test_df=test_dfs[sym], model=models[sym], confidence_threshold=threshold)
    else:
        FastBacktester.run_portfolio_backtest(
            test_dfs=test_dfs,
            models=models,
            initial_capital=capital,
            confidence_threshold=threshold,
        )

    logger.info("=== BACKTEST SIMULATION COMPLETED ===")


def run_live_mode(
    symbols: List[str],
    threshold: float,
    slow_interval: int = 900,
    fast_interval: int = 60,
    run_once: bool = False,
    use_synthetic: bool = False,
    enable_sentiment: bool = True,
) -> None:
    """
    Mode: LIVE (Dual-Speed Paper Trading Engine)
    - Fast Tick (every 60s): manages open positions, checks Stop Loss, Take Profit, Trailing Breakeven.
    - Slow Tick (every 15m / 900s): feature pipeline + XGBoost inference + FinBERT gatekeeper + order execution.
    - If run_once: executes fast tick once + slow tick once across all symbols.
    """
    logger.info("=== INITIALIZING LIVE MULTI-ASSET PAPER TRADING EXECUTION BOT ===")
    models: Dict[str, QuantModel] = {}

    for sym in symbols:
        sym_model_path = get_model_path(sym)
        m = QuantModel(confidence_threshold=threshold, model_path=str(sym_model_path), symbol=sym)
        try:
            m.load_model()
            models[sym] = m
        except Exception as exc:
            logger.error("Model artifact for %s not found (%s). Please run '--mode train' first.", sym, exc)
            sys.exit(1)

    bot = AlpacaExecutionBot(symbols=symbols)
    sentiment_filter = CryptoSentimentFilter() if enable_sentiment else None

    if run_once:
        logger.info("Executing single pulse (--once)...")
        # 1. Fast tick: manage existing open positions
        closed = bot.manage_open_crypto_positions()
        # 2. Slow tick: scan target assets for new entries
        result = bot.run_cycle(
            model=models,
            sentiment_filter=sentiment_filter,
            use_synthetic=use_synthetic,
            symbols=symbols,
        )
        logger.info("Single pulse completed. Result status: %s", result.get("status"))
        return

    logger.info("=" * 65)
    logger.info("=== STARTING DUAL-SPEED LIVE TRADING LOOP ===")
    logger.info("Fast Tick (SL / TP / Trailing Breakeven) : Every %d seconds", fast_interval)
    logger.info("Slow Tick (Signal Discovery / FinBERT)   : Every %d seconds", slow_interval)
    logger.info("Monitored Assets                         : %s", symbols)
    logger.info("=" * 65)

    # Initial scan on launch
    bot.manage_open_crypto_positions()
    bot.run_cycle(model=models, sentiment_filter=sentiment_filter, use_synthetic=use_synthetic, symbols=symbols)
    last_slow_tick = time.time()

    try:
        while True:
            time.sleep(fast_interval)

            # Fast Tick: Position management & Trailing Breakeven
            closed_events = bot.manage_open_crypto_positions()
            if closed_events:
                logger.info("[Fast Tick] Executed exit for %d positions.", len(closed_events))

            # Slow Tick: Check interval for new entry signals
            now = time.time()
            if now - last_slow_tick >= slow_interval:
                logger.info("[Slow Tick] Scanning assets %s for new entry signals...", symbols)
                bot.run_cycle(model=models, sentiment_filter=sentiment_filter, use_synthetic=use_synthetic, symbols=symbols)
                last_slow_tick = now

            # Scheduled Self-Learning Auto-Retrain: Every 7 days
            retrain_sec = RETRAIN_INTERVAL_DAYS * 86400
            if now - bot.last_retrain_time >= retrain_sec:
                logger.info("=== 7-DAY SCHEDULE REACHED: TRIGGERING AUTONOMOUS SELF-LEARNING RETRAINING ===")
                bot.run_self_learning_retrain(
                    models=models,
                    symbols=symbols,
                    lookback_days=LOOKBACK_DAYS,
                    threshold=threshold,
                    use_synthetic=use_synthetic,
                )

    except KeyboardInterrupt:
        logger.info("Trading loop terminated by user.")


def run_status_mode(symbols: List[str]) -> None:
    """
    Mode: STATUS
    Prints portfolio state, active positions across all cryptos, latest real prices,
    and refreshes models/dashboard.html.
    """
    logger.info("=== CHECKING LOCAL PAPER TRADING PORTFOLIO STATUS ===")
    bot = AlpacaExecutionBot(symbols=symbols)
    bot.manage_open_crypto_positions()
    summary = bot.local_broker.get_account_summary()
    positions = bot.local_broker._load_positions()
    bot.local_broker.generate_dashboard_html()

    print("\n" + "=" * 68)
    print("      STANDALONE MULTI-ASSET CRYPTO AGENT — LOCAL PORTFOLIO STATUS       ")
    print("=" * 68)
    prices_list = []
    for s in symbols:
        p = get_latest_price(s)
        prices_list.append(f"{s}: ${p:,.2f}" if p else f"{s}: N/A")
    print(f"Monitored Assets     : {' | '.join(prices_list)}")
    print(f"Total Equity         : ${summary['equity']:,.2f}")
    print(f"Available Cash       : ${summary['cash']:,.2f}")
    print(f"Day Start Equity     : ${summary['day_start_equity']:,.2f}")
    print(f"Today's Return       : {summary['daily_return_pct']:+.2f}%")
    print(f"Unrealized PnL       : ${summary['unrealized_pnl']:+,.2f}")
    print(f"Total Trades Taken   : {summary['total_trades']}")
    print(f"Win Rate             : {summary['win_rate_pct']:.1f}% ({summary['winning_trades']} Wins / {summary['losing_trades']} Losses)")
    print(f"Active Positions     : {len(positions)}")
    print("-" * 68)
    if positions:
        print("ACTIVE POSITIONS:")
        for s, p in positions.items():
            live_price = get_latest_price(s) or p['entry_price']
            unreal = (live_price - p['entry_price']) * p['qty'] if live_price > 0 else 0.0
            unreal_pct = (live_price - p['entry_price']) / p['entry_price'] * 100.0 if live_price > 0 else 0.0
            be_status = "ACTIVE (Risk-Free)" if p.get('breakeven_triggered') else "PENDING"
            print(f"  [{s}] Qty: {p['qty']:.4f} | Entry: ${p['entry_price']:,.2f} | Live: ${live_price:,.2f}")
            print(f"        SL: ${p['stop_loss_price']:,.2f} | TP: ${p['take_profit_price']:,.2f} | Trailing BE: {be_status}")
            print(f"        Unrealized PnL: ${unreal:+,.2f} ({unreal_pct:+.2f}%)")
    else:
        print("No active crypto positions. All funds in Cash.")
    print("=" * 68)
    print(f"Dashboard HTML updated at: {bot.local_broker.dashboard_file}")
    print("=" * 68 + "\n")


def run_retrain_mode(
    symbols: List[str], lookback_days: int, threshold: float, use_synthetic: bool = False
) -> None:
    """
    Mode: RETRAIN
    Executes Self-Learning Retraining across target symbols with Failure Experience Replay
    and Model Quality Gatekeeper, refreshing models/dashboard.html.
    """
    logger.info("=== STARTING AUTONOMOUS SELF-LEARNING RETRAINING PIPELINE ===")
    logger.info("Target Symbols : %s", symbols)
    logger.info("Lookback Days  : %d", lookback_days)
    logger.info("Data Source    : %s", "Synthetic Generator" if use_synthetic else "Yahoo Finance Real Market Data")

    bot = AlpacaExecutionBot(symbols=symbols)
    models: Dict[str, QuantModel] = {}
    for sym in symbols:
        sym_path = get_model_path(sym)
        m = QuantModel(confidence_threshold=threshold, model_path=str(sym_path), symbol=sym)
        models[sym] = m

    results = bot.run_self_learning_retrain(
        models=models,
        symbols=symbols,
        lookback_days=lookback_days,
        threshold=threshold,
        use_synthetic=use_synthetic,
    )

    print("\n" + "=" * 68)
    print("       AUTONOMOUS SELF-LEARNING RETRAINING AUDIT REPORT         ")
    print("=" * 68)
    for sym, r in results.items():
        status = r.get("status", "UNKNOWN")
        reason = r.get("gatekeeper_reason", "")
        metrics = r.get("metrics", {})
        matched = metrics.get("trades_matched", 0)
        penalties = metrics.get("penalties_count", 0)
        boosts = metrics.get("boosts_count", 0)
        roc = metrics.get("test_roc_auc", 0.0)
        prec = metrics.get("confident_precision", 0.0)
        print(f"[{sym}] Status: {status}")
        print(f"       Quality Gatekeeper: {reason}")
        print(f"       ROC-AUC: {roc:.4f} | Confident Precision: {prec:.4f}")
        print(f"       Replayed Trades: {matched} (Penalties 3.0x: {penalties}, Boosts 1.5x: {boosts})")
        print("-" * 68)
    print(f"Dashboard HTML updated at: {bot.local_broker.dashboard_file}")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Quantitative AI Trading Agent (Multi-Asset yfinance & FinBERT ML)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "live", "status", "retrain"],
        required=True,
        help="Execution mode: 'train' model, 'backtest' out-of-sample, 'live' paper trading loop, 'status' check portfolio, or 'retrain' self-learning.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Asset symbol to trade (e.g. BTC-USD, ETH-USD, SOL-USD). Defaults to all assets in SYMBOLS.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK_DAYS,
        help=f"Historical lookback period in days (default: {LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Minimum model win probability threshold for BUY signals (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital for backtesting simulation (default: $100,000)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=900,
        help="Slow tick interval between signal discovery scans in seconds (default: 900s / 15m)",
    )
    parser.add_argument(
        "--fast-interval",
        type=int,
        default=60,
        help="Fast tick interval for active position management in seconds (default: 60s)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single iteration and exit (for cron/scheduled tasks)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicitly use high-fidelity synthetic market data generator (for offline testing)",
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Disable FinBERT news sentiment gatekeeper filter",
    )

    args = parser.parse_args()

    # Resolve target symbols: single symbol if specified, otherwise multi-asset universe
    target_symbols = [args.symbol] if args.symbol and args.symbol.upper() != "ALL" else SYMBOLS

    if args.mode == "train":
        run_training_mode(
            symbols=target_symbols,
            lookback_days=args.lookback,
            threshold=args.threshold,
            use_synthetic=args.synthetic,
        )
    elif args.mode == "backtest":
        run_backtest_mode(
            symbols=target_symbols,
            lookback_days=args.lookback,
            threshold=args.threshold,
            capital=args.capital,
            use_synthetic=args.synthetic,
        )
    elif args.mode == "live":
        run_live_mode(
            symbols=target_symbols,
            threshold=args.threshold,
            slow_interval=args.interval,
            fast_interval=args.fast_interval,
            run_once=args.once,
            use_synthetic=args.synthetic,
            enable_sentiment=not args.no_sentiment,
        )
    elif args.mode == "status":
        run_status_mode(symbols=target_symbols)
    elif args.mode == "retrain":
        run_retrain_mode(
            symbols=target_symbols,
            lookback_days=args.lookback,
            threshold=args.threshold,
            use_synthetic=args.synthetic,
        )


if __name__ == "__main__":
    main()

