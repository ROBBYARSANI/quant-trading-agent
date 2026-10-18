"""
Local Paper Broker Module for Autonomous Quantitative AI Trading Agent.
Provides a 100% Standalone, Persistent, Zero-API-Key Paper Trading Broker
with live real-time price feeds via Yahoo Finance, fee modeling,
client-side bracket/trailing breakeven execution, and auto-generated HTML Dashboard.
"""

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ACTIVE_CRYPTO_POSITIONS_FILE,
    CONFIDENCE_THRESHOLD,
    DASHBOARD_FILE,
    DEFAULT_SLIPPAGE_BPS,
    HORIZON_BARS,
    LOCAL_ACCOUNT_FILE,
    MAX_DAILY_DRAWDOWN,
    RETRAIN_HISTORY_FILE,
    ROOT_DASHBOARD_FILE,
    ROOT_TELEMETRY_FILE,
    SYMBOL,
    SYMBOLS,
    TELEMETRY_FILE,
    TRADE_HISTORY_FILE,
    setup_logger,
)
from data_pipeline import get_latest_price

logger = setup_logger("LocalBroker")


class LocalPaperBroker:
    """
    Persistent, realistic local paper broker simulation engine.
    - Zero external broker API dependencies.
    - Tracks cash, positions, portfolio equity, and intra-day drawdowns.
    - Calculates 0.15% crypto slippage/fees per transaction.
    - Generates a modern HTML dashboard at models/dashboard.html with Live Market Scanner & AI Telemetry.
    """

    def __init__(
        self,
        account_file: Path = LOCAL_ACCOUNT_FILE,
        history_file: Path = TRADE_HISTORY_FILE,
        positions_file: Path = ACTIVE_CRYPTO_POSITIONS_FILE,
        dashboard_file: Path = DASHBOARD_FILE,
        telemetry_file: Path = TELEMETRY_FILE,
        root_dashboard_file: Optional[Path] = ROOT_DASHBOARD_FILE,
        root_telemetry_file: Optional[Path] = ROOT_TELEMETRY_FILE,
        sync_to_root: bool = True,
        fee_bps: float = DEFAULT_SLIPPAGE_BPS,
    ):
        self.account_file = Path(account_file)
        self.history_file = Path(history_file)
        self.positions_file = Path(positions_file)
        self.dashboard_file = Path(dashboard_file)
        self.telemetry_file = Path(telemetry_file)
        self.root_dashboard_file = Path(root_dashboard_file) if root_dashboard_file else None
        self.root_telemetry_file = Path(root_telemetry_file) if root_telemetry_file else None
        # Only sync to workspace root when running against production models/ paths (prevents test pollution)
        is_default_target = (self.dashboard_file.resolve() == DASHBOARD_FILE.resolve())
        self.sync_to_root = bool(sync_to_root and is_default_target)
        self.fee_bps = fee_bps

        self._init_storage()

    def _init_storage(self) -> None:
        """Initializes account JSON, position JSON, trade history CSV, and telemetry JSON if not present."""
        self.account_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.account_file.exists():
            now_iso = datetime.now(timezone.utc).isoformat()
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            initial_state = {
                "cash": 100000.0,
                "equity": 100000.0,
                "day_start_equity": 100000.0,
                "last_reset_date": today_str,
                "created_at": now_iso,
                "updated_at": now_iso,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
            }
            self._save_account(initial_state)

        if not self.positions_file.exists():
            with open(self.positions_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

        if not self.history_file.exists():
            with open(self.history_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "action", "price", "qty",
                    "gross_val", "fee", "net_pnl", "net_pnl_pct", "reason"
                ])

        if not self.telemetry_file.exists():
            default_telemetry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "threshold": CONFIDENCE_THRESHOLD,
                "assets": {
                    s: {
                        "price": 0.0,
                        "win_prob": 0.25,
                        "threshold": CONFIDENCE_THRESHOLD,
                        "signal": "HOLD",
                        "sentiment_score": 0.0,
                        "sentiment_label": "Netral",
                    }
                    for s in SYMBOLS
                },
            }
            with open(self.telemetry_file, "w", encoding="utf-8") as f:
                json.dump(default_telemetry, f, indent=2)

    def _save_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Atomically saves live scanner telemetry state to disk and syncs to root."""
        telemetry["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp_file = self.telemetry_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(telemetry, f, indent=2)
            temp_file.replace(self.telemetry_file)

            # Sync to root folder for static hosting (Hugging Face / GitHub Pages)
            if self.sync_to_root and self.root_telemetry_file:
                try:
                    shutil.copy2(self.telemetry_file, self.root_telemetry_file)
                except Exception as e:
                    logger.debug("Root telemetry sync error: %s", e)
        except Exception as exc:
            logger.debug("Telemetry save error: %s", exc)

    def _load_telemetry(self) -> Dict[str, Any]:
        """Loads latest telemetry state from disk or initializes default state."""
        if self.telemetry_file.exists():
            try:
                with open(self.telemetry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.debug("Telemetry read error: %s", exc)

        default_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold": CONFIDENCE_THRESHOLD,
            "assets": {},
        }
        for s in SYMBOLS:
            live_p = get_latest_price(s) or 0.0
            default_state["assets"][s] = {
                "price": round(live_p, 2),
                "win_prob": 0.25,
                "threshold": CONFIDENCE_THRESHOLD,
                "signal": "HOLD",
                "sentiment_score": 0.0,
                "sentiment_label": "Netral",
            }
        return default_state

    def _load_account(self) -> Dict[str, Any]:
        """Loads account data from disk."""
        try:
            with open(self.account_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Error reading account file (%s): %s", self.account_file, exc)
            return {"cash": 100000.0, "equity": 100000.0, "day_start_equity": 100000.0}

    def _save_account(self, data: Dict[str, Any]) -> None:
        """Saves account data atomically to disk."""
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp_file = self.account_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.account_file)

    def _load_positions(self) -> Dict[str, Any]:
        """Loads active positions from disk."""
        try:
            if self.positions_file.exists():
                with open(self.positions_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            logger.error("Error reading positions file (%s): %s", self.positions_file, exc)
        return {}

    def _save_positions(self, positions: Dict[str, Any]) -> None:
        """Saves active positions atomically to disk."""
        temp_file = self.positions_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(positions, f, indent=2)
        temp_file.replace(self.positions_file)

    def _append_history(
        self,
        symbol: str,
        action: str,
        price: float,
        qty: float,
        gross_val: float,
        fee: float,
        net_pnl: float,
        net_pnl_pct: float,
        reason: str,
    ) -> None:
        """Logs a completed order into CSV history."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.history_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now_str, symbol, action, f"{price:.2f}", f"{qty:.4f}",
                f"{gross_val:.2f}", f"{fee:.2f}", f"{net_pnl:.2f}",
                f"{net_pnl_pct:.2f}%", reason
            ])

    def get_account_summary(self, live_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculates mark-to-market portfolio equity, cash, unrealized PnL,
        and daily drawdown relative to midnight start equity.
        """
        acc = self._load_account()
        positions = self._load_positions()

        # Daily reset check
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if acc.get("last_reset_date") != today_str:
            acc["day_start_equity"] = acc["equity"]
            acc["last_reset_date"] = today_str

        # Calculate position market values
        total_position_value = 0.0
        total_unrealized_pnl = 0.0

        for sym, pos in positions.items():
            entry_price = float(pos["entry_price"])
            qty = float(pos["qty"])
            curr_price = live_price if (live_price and live_price > 0) else get_latest_price(sym)
            if curr_price <= 0:
                curr_price = entry_price

            pos_val = qty * curr_price
            unreal_pnl = (curr_price - entry_price) * qty
            total_position_value += pos_val
            total_unrealized_pnl += unreal_pnl

        equity = acc["cash"] + total_position_value
        acc["equity"] = round(equity, 2)
        self._save_account(acc)

        day_start = float(acc.get("day_start_equity", equity))
        daily_return = (equity - day_start) / day_start if day_start > 0 else 0.0

        total_trades = acc.get("total_trades", 0)
        winning_trades = acc.get("winning_trades", 0)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        return {
            "cash": round(acc["cash"], 2),
            "equity": round(equity, 2),
            "buying_power": round(acc["cash"], 2),
            "day_start_equity": round(day_start, 2),
            "daily_return_pct": round(daily_return * 100.0, 2),
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": acc.get("losing_trades", 0),
            "win_rate_pct": round(win_rate, 2),
            "active_positions_count": len(positions),
        }

    def check_circuit_breaker(self) -> bool:
        """
        Circuit Breaker Guardrail:
        Triggers if intra-day equity drawdown drops <= -MAX_DAILY_DRAWDOWN (-3%).
        """
        summary = self.get_account_summary()
        daily_ret = summary["daily_return_pct"] / 100.0

        if daily_ret <= -MAX_DAILY_DRAWDOWN:
            logger.critical(
                "[CIRCUIT BREAKER TRIGGERED] Intra-day loss of %.2f%% exceeds maximum allowed (-%.2f%%). "
                "Trading halted for remainder of day.",
                daily_ret * 100.0,
                MAX_DAILY_DRAWDOWN * 100.0,
            )
            return True
        return False

    def execute_buy_order(
        self,
        symbol: str,
        qty: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Simulates buy order execution with 0.15% slippage/commission.
        Records active position and writes transaction to trade history.
        """
        acc = self._load_account()
        gross_val = qty * price
        fee = gross_val * self.fee_bps
        total_cost = gross_val + fee

        if acc["cash"] < total_cost:
            logger.warning(
                "Insufficient cash ($%.2f) to buy %.4f %s ($%.2f + $%.2f fee).",
                acc["cash"], qty, symbol, gross_val, fee,
            )
            return None

        acc["cash"] -= total_cost
        self._save_account(acc)

        positions = self._load_positions()
        positions[symbol] = {
            "symbol": symbol,
            "entry_price": round(price, 2),
            "qty": qty,
            "stop_loss_price": round(stop_loss, 2),
            "take_profit_price": round(take_profit, 2),
            "atr": round(atr, 2),
            "entry_timestamp": datetime.now(timezone.utc).isoformat(),
            "max_hold_hours": HORIZON_BARS,
            "breakeven_triggered": False,
        }
        self._save_positions(positions)

        self._append_history(
            symbol=symbol,
            action="BUY",
            price=price,
            qty=qty,
            gross_val=gross_val,
            fee=fee,
            net_pnl=0.0,
            net_pnl_pct=0.0,
            reason="SIGNAL_ENTRY",
        )

        logger.info(
            "[LOCAL BROKER BUY] Executed: %.4f %s @ $%.2f | Total: $%.2f (Fee: $%.2f) | Cash Remaining: $%.2f",
            qty, symbol, price, total_cost, fee, acc["cash"],
        )
        self.generate_dashboard_html()
        return {"status": "FILLED", "action": "BUY", "symbol": symbol, "qty": qty, "price": price}

    def execute_sell_order(
        self, symbol: str, price: float, reason: str
    ) -> Optional[Dict[str, Any]]:
        """
        Simulates sell order execution with 0.15% slippage/commission.
        Calculates realized PnL and updates trade history.
        """
        positions = self._load_positions()
        if symbol not in positions:
            logger.warning("No active position for %s found to close.", symbol)
            return None

        pos = positions[symbol]
        entry_price = float(pos["entry_price"])
        qty = float(pos["qty"])

        gross_val = qty * price
        exit_fee = gross_val * self.fee_bps
        entry_fee = (qty * entry_price) * self.fee_bps
        net_proceeds = gross_val - exit_fee
        net_pnl = (price - entry_price) * qty - exit_fee - entry_fee
        net_pnl_pct = (price - entry_price) / entry_price * 100.0

        # Update Account Cash
        acc = self._load_account()
        acc["cash"] += net_proceeds
        acc["total_trades"] = acc.get("total_trades", 0) + 1
        if net_pnl > 0:
            acc["winning_trades"] = acc.get("winning_trades", 0) + 1
        else:
            acc["losing_trades"] = acc.get("losing_trades", 0) + 1
        self._save_account(acc)

        # Remove from active positions
        del positions[symbol]
        self._save_positions(positions)

        # Record trade
        action_name = f"SELL - {reason}"
        self._append_history(
            symbol=symbol,
            action=action_name,
            price=price,
            qty=qty,
            gross_val=gross_val,
            fee=exit_fee,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            reason=reason,
        )

        logger.info(
            "[LOCAL BROKER SELL] Closed %s (%s): %.4f units @ $%.2f | Net PnL: $%+.2f (%+.2f%%) | New Cash: $%.2f",
            symbol, reason, qty, price, net_pnl, net_pnl_pct, acc["cash"],
        )
        self.generate_dashboard_html()
        return {
            "status": "FILLED",
            "action": action_name,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "net_pnl": round(net_pnl, 2),
            "net_pnl_pct": round(net_pnl_pct, 2),
            "reason": reason,
        }

    def manage_open_positions(self, current_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Client-Side Barrier & Trailing Breakeven Manager:
        Evaluates Stop Loss, Take Profit, Time Barrier, and Trailing Breakeven.
        """
        positions = self._load_positions()
        if not positions:
            return []

        closed_events: List[Dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc)
        updated = dict(positions)

        for sym, pos in positions.items():
            entry_price = float(pos["entry_price"])
            qty = float(pos["qty"])
            sl_price = float(pos["stop_loss_price"])
            tp_price = float(pos["take_profit_price"])
            atr = float(pos.get("atr", (tp_price - entry_price) / 3.0))
            max_hold = int(pos.get("max_hold_hours", HORIZON_BARS))
            breakeven_triggered = bool(pos.get("breakeven_triggered", False))

            # Entry time elapsed
            entry_ts_str = pos.get("entry_timestamp")
            try:
                entry_dt = datetime.fromisoformat(entry_ts_str.replace("Z", "+00:00"))
            except Exception:
                entry_dt = now_utc
            elapsed_hours = (now_utc - entry_dt).total_seconds() / 3600.0

            # Real market price check (handles single test price, multi-asset dict, or live yfinance feed)
            if isinstance(current_price, dict):
                eval_price = float(current_price.get(sym, get_latest_price(sym) or entry_price))
            elif current_price and current_price > 0 and len(positions) == 1:
                eval_price = float(current_price)
            else:
                eval_price = get_latest_price(sym)
            if not eval_price or eval_price <= 0:
                eval_price = entry_price

            # 1. Trailing Breakeven: Price >= Entry + 1.5 * ATR (halfway to TP)
            halfway_mark = entry_price + (1.5 * atr)
            if not breakeven_triggered and eval_price >= halfway_mark and sl_price < entry_price:
                logger.info(
                    "[TRAILING BREAKEVEN] %s reached $%.2f (>= $%.2f). Shifting SL to Entry ($%.2f).",
                    sym, eval_price, halfway_mark, entry_price,
                )
                updated[sym]["stop_loss_price"] = entry_price
                updated[sym]["breakeven_triggered"] = True
                self._save_positions(updated)
                sl_price = entry_price

            # 2. Exit Triggers
            exit_reason = None
            if eval_price <= sl_price:
                exit_reason = "STOP_LOSS"
            elif eval_price >= tp_price:
                exit_reason = "TAKE_PROFIT"
            elif elapsed_hours >= max_hold:
                exit_reason = "TIME_BARRIER"

            if exit_reason:
                close_res = self.execute_sell_order(sym, eval_price, exit_reason)
                if close_res:
                    closed_events.append(close_res)

        self.generate_dashboard_html()
        return closed_events

    def generate_dashboard_html(self, telemetry: Optional[Dict[str, Any]] = None) -> None:
        """
        Renders a state-of-the-art, modern dark-mode HTML dashboard to models/dashboard.html
        including the Live Market Scanner & AI Telemetry panel (BTC, ETH, SOL).
        """
        if telemetry is not None:
            self._save_telemetry(telemetry)

        summary = self.get_account_summary()
        positions = self._load_positions()
        telemetry_data = telemetry or self._load_telemetry()
        assets_telemetry = telemetry_data.get("assets", {})

        # Read last 15 trade history rows
        history_rows: List[List[str]] = []
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    if len(reader) > 1:
                        # Skip header, get latest 15 trades reversed
                        history_rows = reader[1:][-15:][::-1]
            except Exception as e:
                logger.debug("History read error: %s", e)

        # Read latest self-learning retrain audit log
        retrain_display = "Active Calibration (Self-Learning Ready)"
        retrain_badge = "ACTIVE (0 Penalties)"
        if RETRAIN_HISTORY_FILE.exists():
            try:
                with open(RETRAIN_HISTORY_FILE, "r", encoding="utf-8") as f:
                    r_data = json.load(f)
                    if isinstance(r_data, list) and r_data:
                        last_r = r_data[-1]
                        r_ts = last_r.get("timestamp", "")
                        r_dt = r_ts.replace("T", " ")[:16] if r_ts else "Recent"
                        r_stat = last_r.get("status", "ACCEPTED")
                        r_exp = last_r.get("experience_replay", {})
                        r_matched = r_exp.get("trades_matched", 0)
                        r_penalties = r_exp.get("penalties_3x", 0)
                        retrain_display = f"Last Retrain: {r_dt} UTC ({r_stat} &bull; {r_matched} Trades Replayed)"
                        retrain_badge = f"{r_stat} ({r_dt} UTC &bull; {r_penalties} Penalties)"
            except Exception as exc:
                logger.debug("Retrain history read error: %s", exc)

        # Prepare Live Market Scanner & AI Telemetry cards HTML
        scanner_cards_html = ""
        for sym in SYMBOLS:
            t_info = assets_telemetry.get(sym, {})
            price = float(t_info.get("price", 0.0))
            if price <= 0:
                price = float(get_latest_price(sym) or 0.0)

            win_prob = float(t_info.get("win_prob", 0.25))
            win_prob_pct = win_prob * 100.0
            thresh = float(t_info.get("threshold", CONFIDENCE_THRESHOLD))
            signal = str(t_info.get("signal", "HOLD")).upper()
            sent_score = float(t_info.get("sentiment_score", 0.0))
            sent_label = str(t_info.get("sentiment_label", "Netral"))

            bar_class = "above" if win_prob >= thresh else "below"
            bar_width = min(max(win_prob_pct, 5.0), 100.0)
            badge_class = "badge-buy" if signal == "BUY" else "badge-neutral"
            sent_color = "#10b981" if sent_score > 0.15 else ("#ef4444" if sent_score < -0.15 else "#94a3b8")

            # Position check
            is_active = sym in positions
            gate_status = "IN POSITION (MONITORED)" if is_active else ("READY TO BUY" if signal == "BUY" else "HOLD (PRESERVING CASH)")
            gate_color = "#10b981" if is_active or signal == "BUY" else "#94a3b8"

            scanner_cards_html += f"""
            <div class="scanner-card" id="card-{sym}">
                <div class="scanner-header">
                    <span class="scanner-symbol">{sym}</span>
                    <span class="badge {badge_class}" id="badge-{sym}">{signal}</span>
                </div>
                <div class="scanner-price" id="price-{sym}">${price:,.2f}</div>

                <div class="telemetry-row">
                    <span>XGBoost Win Prob</span>
                    <span class="telemetry-val" id="prob-{sym}">{win_prob_pct:.1f}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar {bar_class}" id="bar-{sym}" style="width: {bar_width}%;"></div>
                </div>

                <div class="telemetry-row">
                    <span>FinBERT Sentiment</span>
                    <span class="telemetry-val" style="color: {sent_color};" id="sent-{sym}">
                        {sent_score:+.2f} ({sent_label})
                    </span>
                </div>
                <div class="telemetry-row">
                    <span>System State</span>
                    <span class="telemetry-val" style="color: {gate_color};" id="gate-{sym}">{gate_status}</span>
                </div>
            </div>
            """

        # Prepare active positions rows HTML
        pos_html = ""
        if positions:
            for sym, p in positions.items():
                cur_price = get_latest_price(sym) or p["entry_price"]
                unreal_pnl = (cur_price - p["entry_price"]) * p["qty"]
                unreal_pct = (cur_price - p["entry_price"]) / p["entry_price"] * 100.0
                pnl_color = "#10b981" if unreal_pnl >= 0 else "#ef4444"
                be_badge = (
                    '<span class="badge badge-success">ACTIVE (Risk-Free)</span>'
                    if p.get("breakeven_triggered")
                    else '<span class="badge badge-neutral">PENDING</span>'
                )

                pos_html += f"""
                <tr>
                    <td><strong>{sym}</strong></td>
                    <td>{p['qty']:.4f}</td>
                    <td>${p['entry_price']:,.2f}</td>
                    <td>${cur_price:,.2f}</td>
                    <td>${p['stop_loss_price']:,.2f}</td>
                    <td>${p['take_profit_price']:,.2f}</td>
                    <td>{be_badge}</td>
                    <td style="color: {pnl_color}; font-weight: bold;">
                        {unreal_pnl:+,.2f} ({unreal_pct:+.2f}%)
                    </td>
                </tr>
                """
        else:
            pos_html = '<tr><td colspan="8" style="text-align: center; color: #64748b; padding: 24px;">No active crypto positions. Cash reserves ready.</td></tr>'

        # Prepare trade history rows HTML
        hist_html = ""
        if history_rows:
            for row in history_rows:
                ts, sym, action, price, qty, val, fee, pnl_str, pnl_pct_str, reason = (row + [""] * 10)[:10]
                is_win = not pnl_str.startswith("-") and pnl_str != "0.00"
                pnl_color = "#10b981" if is_win else ("#ef4444" if pnl_str.startswith("-") else "#94a3b8")
                hist_html += f"""
                <tr>
                    <td>{ts}</td>
                    <td><strong>{sym}</strong></td>
                    <td><span class="badge {'badge-buy' if 'BUY' in action else 'badge-sell'}">{action}</span></td>
                    <td>${float(price or 0):,.2f}</td>
                    <td>{float(qty or 0):.4f}</td>
                    <td>${float(val or 0):,.2f}</td>
                    <td>${float(fee or 0):,.2f}</td>
                    <td style="color: {pnl_color}; font-weight: bold;">{pnl_str} ({pnl_pct_str})</td>
                    <td><span class="badge badge-neutral">{reason}</span></td>
                </tr>
                """
        else:
            hist_html = '<tr><td colspan="9" style="text-align: center; color: #64748b; padding: 24px;">No trades recorded yet.</td></tr>'

        updated_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        daily_color = "#10b981" if summary["daily_return_pct"] >= 0 else "#ef4444"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Autonomous Quant Trading Agent - Live Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #090d16;
            --card-bg: rgba(19, 26, 42, 0.75);
            --border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            background-image: radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                              radial-gradient(circle at 90% 90%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
            color: var(--text-primary);
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
            padding: 32px 24px;
        }}
        .container {{ max-width: 1300px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 32px;
        }}
        .brand {{ display: flex; align-items: center; gap: 12px; }}
        .pulse-dot {{
            width: 12px; height: 12px;
            background: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 12px var(--success);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}
        h1 {{ font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }}
        .meta-text {{ color: var(--text-secondary); font-size: 13px; font-family: 'JetBrains Mono', monospace; }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 20px;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .card:hover {{ border-color: rgba(56, 189, 248, 0.3); transform: translateY(-2px); }}
        .card-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.5px; }}
        .card-value {{ font-size: 26px; font-weight: 800; margin-top: 8px; font-family: 'JetBrains Mono', monospace; }}
        .card-sub {{ font-size: 12px; margin-top: 6px; color: var(--text-secondary); }}

        /* Live Market Scanner & AI Telemetry Styles */
        .scanner-panel {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 32px;
        }}
        .scanner-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .scanner-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .scanner-card:hover {{
            border-color: rgba(56, 189, 248, 0.3);
            transform: translateY(-2px);
        }}
        .scanner-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .scanner-symbol {{
            font-size: 15px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }}
        .scanner-price {{
            font-size: 22px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent);
            margin-bottom: 12px;
        }}
        .prob-bar-container {{
            background: rgba(255, 255, 255, 0.06);
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
            margin: 6px 0 10px 0;
            position: relative;
        }}
        .prob-bar {{
            height: 100%;
            border-radius: 6px;
            transition: width 0.4s ease;
        }}
        .prob-bar.above {{
            background: linear-gradient(90deg, #10b981, #34d399);
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
        }}
        .prob-bar.below {{
            background: linear-gradient(90deg, #64748b, #94a3b8);
        }}
        .telemetry-row {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 6px;
        }}
        .telemetry-val {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .section-title {{
            font-size: 18px; font-weight: 700; margin-bottom: 16px;
            display: flex; align-items: center; justify-content: space-between;
        }}
        .table-wrap {{
            overflow-x: auto;
            border-radius: 16px;
            border: 1px solid var(--border);
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            margin-bottom: 32px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background: rgba(255, 255, 255, 0.03);
            padding: 14px 18px;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 14px 18px;
            border-bottom: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
        }}
        tr:last-child td {{ border-bottom: none; }}
        .badge {{
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }}
        .badge-buy {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-sell {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge-success {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
        .badge-neutral {{ background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid var(--border); }}
        footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <div class="pulse-dot"></div>
                <div>
                    <h1>Autonomous Quant Trading Agent</h1>
                    <div class="meta-text">Standalone Indonesia-Ready Crypto Paper Broker & Machine Learning Engine</div>
                </div>
            </div>
            <div style="text-align: right;">
                <div class="meta-text">LAST REFRESH: {updated_time}</div>
                <div class="meta-text" style="color: #a78bfa; margin-top: 4px;">SELF-LEARNING AI: {retrain_badge}</div>
                <div class="meta-text" style="color: var(--success); margin-top: 4px;">CIRCUIT BREAKER: ARMED (-3.0%)</div>
            </div>
        </header>

        <div class="grid-stats">
            <div class="card">
                <div class="card-label">Total Portfolio Equity</div>
                <div class="card-value">${summary['equity']:,.2f}</div>
                <div class="card-sub">Cash: ${summary['cash']:,.2f}</div>
            </div>
            <div class="card">
                <div class="card-label">Today's Performance</div>
                <div class="card-value" style="color: {daily_color};">{summary['daily_return_pct']:+.2f}%</div>
                <div class="card-sub">Day Start: ${summary['day_start_equity']:,.2f}</div>
            </div>
            <div class="card">
                <div class="card-label">Unrealized PnL</div>
                <div class="card-value" style="color: {'#10b981' if summary['unrealized_pnl'] >= 0 else '#ef4444'};">
                    ${summary['unrealized_pnl']:+,.2f}
                </div>
                <div class="card-sub">Active Positions: {summary['active_positions_count']}</div>
            </div>
            <div class="card">
                <div class="card-label">Win Rate / Executions</div>
                <div class="card-value">{summary['win_rate_pct']:.1f}%</div>
                <div class="card-sub">{summary['winning_trades']} Wins / {summary['total_trades']} Total Trades</div>
            </div>
        </div>

        <!-- Live Market Scanner & AI Telemetry Panel -->
        <div class="scanner-panel">
            <div class="section-title" style="margin-bottom: 0;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div class="pulse-dot" style="width: 8px; height: 8px; background: #38bdf8; box-shadow: 0 0 8px #38bdf8;"></div>
                    <span>Live Market Scanner &amp; AI Telemetry (BTC, ETH, SOL)</span>
                </div>
                <span class="meta-text" id="telemetry-time">Thresh: 54% | Half-Kelly | {retrain_display}</span>
            </div>
            <div class="scanner-grid" id="scanner-grid">
                {scanner_cards_html}
            </div>
        </div>

        <div class="section-title">
            <span>Active Cryptocurrency Positions</span>
            <span class="meta-text">{len(positions)} Tracked</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Position Qty</th>
                        <th>Entry Price</th>
                        <th>Live Price</th>
                        <th>Stop Loss</th>
                        <th>Take Profit</th>
                        <th>Trailing Breakeven</th>
                        <th>Unrealized Return</th>
                    </tr>
                </thead>
                <tbody>
                    {pos_html}
                </tbody>
            </table>
        </div>

        <div class="section-title">
            <span>Transaction Ledger & Execution History</span>
            <span class="meta-text">Last 15 Executions</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Timestamp (UTC)</th>
                        <th>Symbol</th>
                        <th>Order Action</th>
                        <th>Execution Price</th>
                        <th>Units</th>
                        <th>Gross Value</th>
                        <th>Fee (0.15%)</th>
                        <th>Realized PnL</th>
                        <th>Exit Barrier Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {hist_html}
                </tbody>
            </table>
        </div>

        <footer>
            Autonomous Quant Trading Agent &bull; Triple-Barrier XGBoost &bull; Fractional Kelly &bull; FinBERT NLP &bull; Zero External API Key Required
        </footer>
    </div>

    <script>
        async function fetchTelemetry() {{
            try {{
                const res = await fetch('latest_telemetry.json?nocache=' + Date.now());
                if (!res.ok) return;
                const data = await res.json();
                if (!data || !data.assets) return;

                for (const [sym, info] of Object.entries(data.assets)) {{
                    const priceEl = document.getElementById('price-' + sym);
                    const probEl = document.getElementById('prob-' + sym);
                    const barEl = document.getElementById('bar-' + sym);
                    const badgeEl = document.getElementById('badge-' + sym);
                    const sentEl = document.getElementById('sent-' + sym);

                    if (priceEl && info.price) {{
                        priceEl.textContent = '$' + Number(info.price).toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                    }}
                    if (probEl && info.win_prob !== undefined) {{
                        const probPct = (info.win_prob * 100).toFixed(1);
                        probEl.textContent = probPct + '%';
                        if (barEl) {{
                            barEl.style.width = Math.min(Math.max(probPct, 5), 100) + '%';
                            if (info.win_prob >= (info.threshold || 0.54)) {{
                                barEl.className = 'prob-bar above';
                            }} else {{
                                barEl.className = 'prob-bar below';
                            }}
                        }}
                    }}
                    if (badgeEl && info.signal) {{
                        badgeEl.textContent = info.signal;
                        badgeEl.className = 'badge ' + (info.signal === 'BUY' ? 'badge-buy' : 'badge-neutral');
                    }}
                    if (sentEl && info.sentiment_score !== undefined) {{
                        const score = Number(info.sentiment_score);
                        const label = info.sentiment_label || (score > 0.15 ? 'Positif' : (score < -0.15 ? 'Negatif' : 'Netral'));
                        sentEl.textContent = (score >= 0 ? '+' : '') + score.toFixed(2) + ' (' + label + ')';
                        sentEl.style.color = score > 0.15 ? '#10b981' : (score < -0.15 ? '#ef4444' : '#94a3b8');
                    }}
                }}
                const timeEl = document.getElementById('telemetry-time');
                if (timeEl && data.timestamp) {{
                    timeEl.textContent = 'Last Scan: ' + new Date(data.timestamp).toLocaleTimeString() + ' | Threshold: 54%';
                }}
            }} catch (err) {{
                console.debug('Telemetry auto-refresh error:', err);
            }}
        }}
        setInterval(fetchTelemetry, 30000);
    </script>
</body>
</html>
"""
        try:
            with open(self.dashboard_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info("Dashboard HTML updated at %s", self.dashboard_file)

            # Sync to root index.html for static hosting (Hugging Face / GitHub Pages)
            if self.sync_to_root and self.root_dashboard_file:
                try:
                    shutil.copy2(self.dashboard_file, self.root_dashboard_file)
                    logger.info("Synced dashboard to root: %s", self.root_dashboard_file)
                except Exception as e:
                    logger.debug("Root dashboard sync error: %s", e)
        except Exception as exc:
            logger.error("Failed to generate dashboard HTML: %s", exc)
