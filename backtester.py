"""
Backtester Module for Autonomous Quantitative AI Trading Agent.
Executes an event-driven, out-of-sample backtest simulation with slippage/commission modeling,
calculating institutional-grade quantitative risk and performance metrics.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import (
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    CONFIDENCE_THRESHOLD,
    DEFAULT_SLIPPAGE_BPS,
    HORIZON_BARS,
    MAX_POSITION_EXPOSURE,
    MAX_RISK_PER_TRADE,
    SYMBOL,
    is_crypto_symbol,
    setup_logger,
)
from model_engine import QuantModel
from risk_allocator import calculate_optimal_allocation

logger = setup_logger("Backtester")


class FastBacktester:
    """
    Simulates high-fidelity execution of the trained QuantModel strategy
    on strictly out-of-sample (unseen) historical market data.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        symbol: str = SYMBOL,
    ):
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.symbol = symbol
        self.is_crypto = is_crypto_symbol(symbol)
        # Annualization factor: 24*365=8760 hours for crypto, ~252*7=1764 hours for equities
        self.annual_bars = 8760 if self.is_crypto else 1764

    def run_backtest(
        self,
        test_df: pd.DataFrame,
        model: QuantModel,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Executes sequential bar-by-bar backtest over test_df.
        """
        logger.info(
            "Starting Out-of-Sample Backtest on %d bars with $%s initial equity...",
            len(test_df),
            f"{self.initial_capital:,.2f}",
        )

        equity = self.initial_capital
        buying_power = equity
        peak_equity = equity

        equity_curve: List[float] = [equity]
        trade_log: List[Dict[str, Any]] = []

        # Current trade state
        in_position = False
        entry_idx = 0
        entry_price = 0.0
        qty = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        atr_at_entry = 0.0
        breakeven_triggered = False
        bars_held = 0

        timestamps = test_df.index
        closes = test_df["close"].values
        highs = test_df["high"].values
        lows = test_df["low"].values
        atrs = test_df["atr"].values

        # Precompute model predictions across test_df for speed
        feature_data = test_df[model.feature_columns].values
        win_probs = model.model.predict_proba(feature_data)[:, 1]

        n = len(test_df)

        for i in range(n):
            curr_time = timestamps[i]
            curr_close = closes[i]
            curr_high = highs[i]
            curr_low = lows[i]
            curr_atr = atrs[i]

            # 1. Manage Active Position
            if in_position:
                bars_held += 1
                exit_price = None
                exit_reason = None

                # Trailing Breakeven check: Price >= Entry + 1.5 * ATR
                halfway_mark = entry_price + (1.5 * atr_at_entry)
                if not breakeven_triggered and curr_high >= halfway_mark and stop_loss < entry_price:
                    stop_loss = entry_price
                    breakeven_triggered = True

                # Check Take Profit
                if curr_high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TAKE_PROFIT"
                # Check Stop Loss
                elif curr_low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "STOP_LOSS (BREAKEVEN)" if breakeven_triggered and stop_loss >= entry_price else "STOP_LOSS"
                # Check Time Horizon
                elif bars_held >= HORIZON_BARS:
                    exit_price = curr_close
                    exit_reason = "TIME_EXPIRY"

                if exit_price is not None:
                    # Account for exit slippage
                    slippage_exit = exit_price * qty * self.slippage_bps
                    gross_pnl = (exit_price - entry_price) * qty
                    net_pnl = gross_pnl - slippage_exit

                    equity += net_pnl
                    buying_power = equity
                    in_position = False

                    trade_record = {
                        "symbol": self.symbol,
                        "entry_time": timestamps[entry_idx],
                        "exit_time": curr_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "bars_held": bars_held,
                        "gross_pnl": gross_pnl,
                        "slippage": slippage_exit,
                        "net_pnl": net_pnl,
                        "return_pct": (exit_price - entry_price) / entry_price * 100.0,
                        "exit_reason": exit_reason,
                        "win": 1 if net_pnl > 0 else 0,
                    }
                    trade_log.append(trade_record)

            # 2. Check for New Entry Signal (if not currently in position)
            if not in_position and i < n - 1:
                prob = win_probs[i]
                if prob >= confidence_threshold:
                    alloc = calculate_optimal_allocation(
                        equity=equity,
                        buying_power=buying_power,
                        entry_price=curr_close,
                        atr=curr_atr,
                        win_prob=prob,
                        symbol=self.symbol,
                        max_exposure_pct=MAX_POSITION_EXPOSURE,
                    )
                    trade_qty = alloc.get("qty", 0.0)

                    if trade_qty > 0 and alloc.get("status") == "APPROVED":
                        in_position = True
                        entry_idx = i
                        entry_price = curr_close
                        qty = trade_qty
                        stop_loss = alloc["stop_loss_price"]
                        take_profit = alloc["take_profit_price"]
                        atr_at_entry = curr_atr
                        breakeven_triggered = False
                        bars_held = 0

                        # Deduct entry slippage
                        slippage_entry = entry_price * qty * self.slippage_bps
                        equity -= slippage_entry

            # Track equity mark-to-market
            if in_position:
                unrealized_pnl = (curr_close - entry_price) * qty
                current_val = equity + unrealized_pnl
            else:
                current_val = equity

            equity_curve.append(current_val)
            if current_val > peak_equity:
                peak_equity = current_val

        # Compile Institutional Metrics
        metrics = self._calculate_metrics(
            initial_equity=self.initial_capital,
            equity_curve=np.array(equity_curve),
            trade_log=trade_log,
            benchmark_start=closes[0],
            benchmark_end=closes[-1],
        )

        self._print_backtest_report(metrics)
        return metrics

    def _calculate_metrics(
        self,
        initial_equity: float,
        equity_curve: np.ndarray,
        trade_log: List[Dict[str, Any]],
        benchmark_start: float,
        benchmark_end: float,
    ) -> Dict[str, Any]:
        """Calculates performance and risk analytics."""
        ending_equity = equity_curve[-1]
        total_return_pct = ((ending_equity - initial_equity) / initial_equity) * 100.0
        benchmark_return_pct = ((benchmark_end - benchmark_start) / benchmark_start) * 100.0

        # Drawdown computation
        peak_series = np.maximum.accumulate(equity_curve)
        drawdowns = (peak_series - equity_curve) / peak_series
        max_drawdown_pct = float(np.max(drawdowns)) * 100.0

        # Bar-to-bar returns for Sharpe Ratio
        bar_returns = np.diff(equity_curve) / equity_curve[:-1]
        std_ret = np.std(bar_returns)
        mean_ret = np.mean(bar_returns)

        if std_ret > 1e-8:
            sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(self.annual_bars))
        else:
            sharpe_ratio = 0.0

        # Trade analytics
        total_trades = len(trade_log)
        if total_trades > 0:
            net_pnls = [t["net_pnl"] for t in trade_log]
            winning_trades = [p for p in net_pnls if p > 0]
            losing_trades = [p for p in net_pnls if p <= 0]

            win_rate_pct = (len(winning_trades) / total_trades) * 100.0
            gross_gains = sum(winning_trades)
            gross_losses = abs(sum(losing_trades))
            profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else float("inf")
            avg_trade_pnl = np.mean(net_pnls)
        else:
            win_rate_pct = 0.0
            profit_factor = 0.0
            avg_trade_pnl = 0.0
            winning_trades = []
            losing_trades = []

        return {
            "initial_capital": initial_equity,
            "ending_equity": ending_equity,
            "total_return_pct": total_return_pct,
            "benchmark_return_pct": benchmark_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": win_rate_pct,
            "profit_factor": profit_factor,
            "avg_trade_pnl": float(avg_trade_pnl),
            "trades": trade_log,
        }

    def _print_backtest_report(self, m: Dict[str, Any]) -> None:
        """Prints a professional institutional report."""
        logger.info("=" * 65)
        logger.info("       QUANTITATIVE AGENT OUT-OF-SAMPLE BACKTEST REPORT       ")
        logger.info("=" * 65)
        logger.info("Initial Capital     : $%s", f"{m['initial_capital']:,.2f}")
        logger.info("Ending Equity       : $%s", f"{m['ending_equity']:,.2f}")
        logger.info("Strategy Return     : %+.2f%%", m["total_return_pct"])
        logger.info("Buy & Hold Return   : %+.2f%%", m["benchmark_return_pct"])
        logger.info("Annualized Sharpe   : %.2f", m["sharpe_ratio"])
        logger.info("Maximum Drawdown    : %.2f%%", m["max_drawdown_pct"])
        logger.info("-" * 65)
        logger.info("Total Trades Taken  : %d", m["total_trades"])
        logger.info("Winning Trades      : %d (%.1f%%)", m["winning_trades"], m["win_rate_pct"])
        logger.info("Losing Trades       : %d", m["losing_trades"])
        logger.info("Profit Factor       : %.2f", m["profit_factor"])
        logger.info("Average Trade Net   : $%+.2f", m["avg_trade_pnl"])
        logger.info("=" * 65)

    @classmethod
    def run_portfolio_backtest(
        cls,
        test_dfs: Dict[str, pd.DataFrame],
        models: Dict[str, QuantModel],
        initial_capital: float = 100_000.0,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Executes unified portfolio backtest simulation across multiple crypto assets (BTC, ETH, SOL).
        1. Backtests each asset independently (standalone analysis).
        2. Simulates synchronized event-driven portfolio sharing $100,000 capital.
        3. Generates combined institutional metrics.
        """
        logger.info("=" * 65)
        logger.info("   STARTING MULTI-ASSET CRYPTO PORTFOLIO BACKTEST SIMULATION   ")
        logger.info("   Assets: %s | Shared Initial Capital: $%s", list(test_dfs.keys()), f"{initial_capital:,.2f}")
        logger.info("=" * 65)

        # 1. Individual Asset Backtests
        asset_results: Dict[str, Dict[str, Any]] = {}
        for sym, df_sym in test_dfs.items():
            model = models.get(sym) or models.get("default")
            bt = cls(initial_capital=initial_capital, slippage_bps=slippage_bps, symbol=sym)
            logger.info(">>> Backtesting Standalone Asset: %s (%d bars) <<<", sym, len(df_sym))
            res = bt.run_backtest(test_df=df_sym, model=model, confidence_threshold=confidence_threshold)
            asset_results[sym] = res

        # 2. Synchronized Shared-Portfolio Simulation
        common_timestamps = sorted(list(set.intersection(*[set(df.index) for df in test_dfs.values()])))
        if not common_timestamps:
            common_timestamps = sorted(list(set.union(*[set(df.index) for df in test_dfs.values()])))

        cash = initial_capital
        peak_equity = initial_capital
        portfolio_equity_curve = [initial_capital]
        portfolio_trade_log = []
        open_positions: Dict[str, Dict[str, Any]] = {}

        # Precompute win probabilities for each asset
        win_probs_dict = {}
        for sym, df in test_dfs.items():
            m = models.get(sym) or models.get("default")
            feats = df[m.feature_columns].values
            win_probs_dict[sym] = pd.Series(m.model.predict_proba(feats)[:, 1], index=df.index)

        # Equal-weighted benchmark returns
        first_ts = common_timestamps[0]
        last_ts = common_timestamps[-1]
        benchmark_returns = []
        for sym, df in test_dfs.items():
            if first_ts in df.index and last_ts in df.index:
                p0 = float(df.loc[first_ts, "close"])
                p1 = float(df.loc[last_ts, "close"])
                if p0 > 0:
                    benchmark_returns.append((p1 - p0) / p0)
        avg_benchmark_return_pct = (float(np.mean(benchmark_returns)) * 100.0) if benchmark_returns else 0.0

        for t in common_timestamps:
            # A. Manage existing open positions across assets
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                pos["bars_held"] += 1
                df_sym = test_dfs[sym]
                if t not in df_sym.index:
                    continue

                curr_close = float(df_sym.loc[t, "close"])
                curr_high = float(df_sym.loc[t, "high"])
                curr_low = float(df_sym.loc[t, "low"])

                # Trailing Breakeven check
                halfway = pos["entry_price"] + (1.5 * pos["atr"])
                if not pos["breakeven_triggered"] and curr_high >= halfway and pos["stop_loss"] < pos["entry_price"]:
                    pos["stop_loss"] = pos["entry_price"]
                    pos["breakeven_triggered"] = True

                exit_price = None
                exit_reason = None
                if curr_high >= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "TAKE_PROFIT"
                elif curr_low <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "STOP_LOSS (BREAKEVEN)" if pos["breakeven_triggered"] and pos["stop_loss"] >= pos["entry_price"] else "STOP_LOSS"
                elif pos["bars_held"] >= HORIZON_BARS:
                    exit_price = curr_close
                    exit_reason = "TIME_EXPIRY"

                if exit_price is not None:
                    exit_slippage = exit_price * pos["qty"] * slippage_bps
                    gross_pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                    net_pnl = gross_pnl - exit_slippage - pos["entry_slippage"]
                    cash += (exit_price * pos["qty"]) - exit_slippage

                    portfolio_trade_log.append({
                        "symbol": sym,
                        "entry_time": pos["entry_time"],
                        "exit_time": t,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "qty": pos["qty"],
                        "bars_held": pos["bars_held"],
                        "gross_pnl": gross_pnl,
                        "slippage": exit_slippage + pos["entry_slippage"],
                        "net_pnl": net_pnl,
                        "return_pct": (exit_price - pos["entry_price"]) / pos["entry_price"] * 100.0,
                        "exit_reason": exit_reason,
                        "win": 1 if net_pnl > 0 else 0,
                    })
                    del open_positions[sym]

            # B. Check for new entry signals across assets
            current_portfolio_equity = cash + sum(
                open_positions[s]["qty"] * float(test_dfs[s].loc[t, "close"])
                for s in open_positions if t in test_dfs[s].index
            )

            for sym in test_dfs:
                if sym in open_positions:
                    continue
                df_sym = test_dfs[sym]
                if t not in df_sym.index:
                    continue

                prob = float(win_probs_dict[sym].loc[t])
                if prob >= confidence_threshold:
                    curr_close = float(df_sym.loc[t, "close"])
                    curr_atr = float(df_sym.loc[t, "atr"])

                    alloc = calculate_optimal_allocation(
                        equity=current_portfolio_equity,
                        buying_power=cash,
                        entry_price=curr_close,
                        atr=curr_atr,
                        win_prob=prob,
                        symbol=sym,
                        max_exposure_pct=MAX_POSITION_EXPOSURE,
                    )
                    trade_qty = alloc.get("qty", 0.0)
                    cost = trade_qty * curr_close
                    entry_slippage = cost * slippage_bps

                    if trade_qty > 0 and alloc.get("status") == "APPROVED" and cash >= (cost + entry_slippage):
                        cash -= (cost + entry_slippage)
                        open_positions[sym] = {
                            "symbol": sym,
                            "entry_time": t,
                            "entry_price": curr_close,
                            "qty": trade_qty,
                            "stop_loss": alloc["stop_loss_price"],
                            "take_profit": alloc["take_profit_price"],
                            "atr": curr_atr,
                            "bars_held": 0,
                            "entry_slippage": entry_slippage,
                            "breakeven_triggered": False,
                        }

            # C. Record Mark-to-Market Portfolio Equity
            bar_equity = cash + sum(
                open_positions[s]["qty"] * float(test_dfs[s].loc[t, "close"])
                for s in open_positions if t in test_dfs[s].index
            )
            portfolio_equity_curve.append(bar_equity)
            if bar_equity > peak_equity:
                peak_equity = bar_equity

        # Compile Portfolio Metrics
        bt_engine = cls(initial_capital=initial_capital, slippage_bps=slippage_bps, symbol="PORTFOLIO")
        portfolio_metrics = bt_engine._calculate_metrics(
            initial_equity=initial_capital,
            equity_curve=np.array(portfolio_equity_curve),
            trade_log=portfolio_trade_log,
            benchmark_start=1.0,
            benchmark_end=1.0 + (avg_benchmark_return_pct / 100.0),
        )

        # Print Institutional Multi-Asset Summary
        logger.info("=" * 70)
        logger.info("         MULTI-ASSET CRYPTO PORTFOLIO PERFORMANCE SUMMARY       ")
        logger.info("=" * 70)
        logger.info("%-10s | %-12s | %-9s | %-8s | %-7s | %-7s | %-8s",
                    "Asset", "Ending Equity", "Return", "Win Rate", "Trades", "Sharpe", "Max DD")
        logger.info("-" * 70)
        for sym, res in asset_results.items():
            logger.info("%-10s | $%11s | %+8.2f%% | %7.1f%% | %6d | %7.2f | %7.2f%%",
                        sym, f"{res['ending_equity']:,.2f}", res["total_return_pct"],
                        res["win_rate_pct"], res["total_trades"], res["sharpe_ratio"], res["max_drawdown_pct"])
        logger.info("-" * 70)
        logger.info("%-10s | $%11s | %+8.2f%% | %7.1f%% | %6d | %7.2f | %7.2f%%",
                    "COMBINED", f"{portfolio_metrics['ending_equity']:,.2f}", portfolio_metrics["total_return_pct"],
                    portfolio_metrics["win_rate_pct"], portfolio_metrics["total_trades"],
                    portfolio_metrics["sharpe_ratio"], portfolio_metrics["max_drawdown_pct"])
        logger.info("=" * 70)
        logger.info("Combined Portfolio Profit Factor : %.2f", portfolio_metrics["profit_factor"])
        logger.info("Combined Portfolio Average Trade : $%+.2f", portfolio_metrics["avg_trade_pnl"])
        logger.info("Benchmark (Equal-Weighted Crypto): %+.2f%%", portfolio_metrics["benchmark_return_pct"])
        logger.info("=" * 70)

        return {"portfolio": portfolio_metrics, "assets": asset_results}
