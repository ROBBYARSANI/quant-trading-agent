"""
Risk Allocator Module for Autonomous Quantitative AI Trading Agent.
Implements Dynamic Capital Allocation via Fractional Kelly Criterion (Quarter-Kelly)
and ATR Volatility Sizing to determine optimal position size and protective bracket bounds.
"""

from typing import Any, Dict
import numpy as np

from config import (
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    MAX_POSITION_EXPOSURE,
    MAX_RISK_PER_TRADE,
    MIN_RISK_PER_TRADE,
    SYMBOL,
    is_crypto_symbol,
    setup_logger,
)

logger = setup_logger("RiskAllocator")


def calculate_optimal_allocation(
    equity: float,
    buying_power: float,
    entry_price: float,
    atr: float,
    win_prob: float,
    symbol: str = SYMBOL,
    atr_sl_mult: float = ATR_SL_MULTIPLIER,
    atr_tp_mult: float = ATR_TP_MULTIPLIER,
    max_risk_pct: float = MAX_RISK_PER_TRADE,
    min_risk_pct: float = MIN_RISK_PER_TRADE,
    max_exposure_pct: float = MAX_POSITION_EXPOSURE,
) -> Dict[str, Any]:
    """
    Calculates quantitative risk-adjusted order parameters:
    1. Reward-to-Risk ratio: b = ATR_TP_MULTIPLIER / ATR_SL_MULTIPLIER
    2. Quarter-Kelly Fraction: kelly_f = 0.25 * ((win_prob * b - (1 - win_prob)) / b)
    3. Clamps risk_pct between [min_risk_pct (0.25%), max_risk_pct (1.0%)]
    4. Computes ATR-based Stop Loss and Take Profit bounds
    5. Sizes position units: qty = (equity * risk_pct) / (entry_price - stop_loss_price)
    6. Caps total position value at min(equity * max_exposure_pct, buying_power)

    Returns:
        dict with keys: 'qty', 'entry_price', 'stop_loss_price', 'take_profit_price',
                        'risk_amount', 'risk_pct', 'position_value', 'kelly_fraction'
    """
    if equity <= 0:
        logger.error("Invalid equity: %.2f. Capital must be positive.", equity)
        return {"qty": 0.0, "risk_pct": 0.0, "status": "INVALID_EQUITY"}

    if entry_price <= 0 or atr <= 0:
        logger.error("Invalid entry price (%.2f) or ATR (%.2f)", entry_price, atr)
        return {"qty": 0.0, "risk_pct": 0.0, "status": "INVALID_PRICE_OR_ATR"}

    # 1. Odds / Reward-to-Risk ratio
    b = atr_tp_mult / atr_sl_mult  # Typically 3.0 / 1.5 = 2.0

    # 2. Fractional Kelly Criterion (Half Kelly for optimal crypto risk-reward)
    # Kelly Formula: f* = (p * b - q) / b, where q = 1 - p
    loss_prob = 1.0 - win_prob
    raw_kelly = (win_prob * b - loss_prob) / b
    half_kelly = 0.5 * raw_kelly

    # 3. Constrain risk percentage
    if half_kelly <= 0:
        logger.warning(
            "Half Kelly fraction is non-positive (%.4f) for win_prob=%.4f. Trade rejected.",
            half_kelly,
            win_prob,
        )
        risk_pct = 0.0
    else:
        risk_pct = float(np.clip(half_kelly, min_risk_pct, max_risk_pct))

    if risk_pct == 0.0:
        return {
            "qty": 0.0,
            "entry_price": entry_price,
            "stop_loss_price": 0.0,
            "take_profit_price": 0.0,
            "risk_amount": 0.0,
            "risk_pct": 0.0,
            "position_value": 0.0,
            "kelly_fraction": half_kelly,
            "status": "NO_EDGE",
        }

    # 4. Bracket price boundaries (formatted to 2 decimal places)
    stop_distance = atr_sl_mult * atr
    profit_distance = atr_tp_mult * atr

    stop_loss_price = round(entry_price - stop_distance, 2)
    take_profit_price = round(entry_price + profit_distance, 2)

    # Sanity checks on barrier prices
    if stop_loss_price <= 0:
        stop_loss_price = round(entry_price * 0.95, 2)
    risk_per_unit = entry_price - stop_loss_price

    if risk_per_unit <= 0:
        logger.error("Risk per unit is non-positive: %.2f", risk_per_unit)
        return {"qty": 0.0, "risk_pct": 0.0, "status": "INVALID_RISK_PER_UNIT"}

    # 5. Position Sizing
    risk_amount = equity * risk_pct
    raw_qty = risk_amount / risk_per_unit

    # 6. Apply Maximum Position Value & Buying Power Constraints
    max_position_value = min(equity * max_exposure_pct, buying_power * 0.98)
    position_value = raw_qty * entry_price

    if position_value > max_position_value:
        logger.info(
            "Position value $%.2f exceeds ceiling $%.2f. Scaling qty down.",
            position_value,
            max_position_value,
        )
        raw_qty = max_position_value / entry_price
        position_value = raw_qty * entry_price
        risk_amount = raw_qty * risk_per_unit

    # Format quantity precision (Cryptos support fractional units down to 4 decimals; stocks 2 or 0)
    is_crypto = is_crypto_symbol(symbol)
    if is_crypto:
        qty = round(raw_qty, 4)
    else:
        qty = round(raw_qty, 2)

    logger.info(
        "Risk Allocation Calculated: WinProb=%.2f%% | Half-Kelly=%.4f | RiskPct=%.2f%% "
        "| Qty=%.4f ($%.2f) | Entry=$%.2f | SL=$%.2f | TP=$%.2f | MaxRisk=$%.2f",
        win_prob * 100.0,
        half_kelly,
        risk_pct * 100.0,
        qty,
        qty * entry_price,
        entry_price,
        stop_loss_price,
        take_profit_price,
        risk_amount,
    )

    return {
        "qty": qty,
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "risk_amount": round(risk_amount, 2),
        "risk_pct": round(risk_pct, 4),
        "position_value": round(qty * entry_price, 2),
        "kelly_fraction": round(half_kelly, 4),
        "status": "APPROVED",
    }
