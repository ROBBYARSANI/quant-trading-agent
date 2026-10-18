"""
Model Engine Module for Autonomous Quantitative AI Trading Agent.
Implements Marcos Lopez de Prado's Triple-Barrier Labeling method and an XGBoost Classifier
with strictly chronological train/test splitting to prevent time-series data leakage.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, roc_auc_score
from xgboost import XGBClassifier

from config import (
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    CONFIDENCE_THRESHOLD,
    EXPERIENCE_REPLAY_WINDOW_HOURS,
    FAILURE_SAMPLE_WEIGHT,
    HORIZON_BARS,
    MODEL_PATH,
    RETRAIN_HISTORY_FILE,
    RETRAIN_INTERVAL_DAYS,
    SUCCESS_SAMPLE_WEIGHT,
    SYMBOL,
    TRADE_HISTORY_FILE,
    get_model_path,
    setup_logger,
)
from data_pipeline import STATIONARY_FEATURE_COLUMNS

logger = setup_logger("ModelEngine")


def _append_retrain_history(
    record: Dict[str, Any], history_file: Path = RETRAIN_HISTORY_FILE
) -> None:
    """Appends model retraining audit log to retrain_history.json."""
    hist_path = Path(history_file)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    if hist_path.exists():
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception as exc:
            logger.warning("Could not parse existing retrain history: %s", exc)
            history = []

    history.append(record)
    temp_path = hist_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        temp_path.replace(hist_path)
    except Exception as exc:
        logger.error("Failed to append retrain history: %s", exc)


def load_retrain_history(
    history_file: Path = RETRAIN_HISTORY_FILE,
) -> List[Dict[str, Any]]:
    """Loads all model retraining audit records."""
    hist_path = Path(history_file)
    if hist_path.exists():
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as exc:
            logger.debug("Error loading retrain history: %s", exc)
    return []


def extract_trade_feedback(
    trade_history: Optional[Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]] = None,
    symbol: str = SYMBOL,
) -> List[Dict[str, Any]]:
    """
    Parses trade history records to extract completed trade outcomes (entry time, exit time, net PnL).
    Pairs BUY and SELL actions to identify the exact entry timestamp.
    """
    clean_target = symbol.replace("/", "-").upper()
    completed_trades: List[Dict[str, Any]] = []

    # Case 1: Direct List of Dictionaries
    if isinstance(trade_history, list):
        for item in trade_history:
            sym = str(item.get("symbol", "")).replace("/", "-").upper()
            if not sym or sym == clean_target:
                completed_trades.append(item)
        return completed_trades

    # Case 2: Direct DataFrame
    if isinstance(trade_history, pd.DataFrame):
        df_hist = trade_history.copy()
        if "action" in df_hist.columns:
            open_buys: List[Dict[str, Any]] = []
            for _, row in df_hist.iterrows():
                sym = str(row.get("symbol", "")).replace("/", "-").upper()
                if sym != clean_target and sym:
                    continue
                action = str(row.get("action", "")).upper()
                if action == "BUY":
                    open_buys.append(dict(row))
                elif action.startswith("SELL"):
                    entry_ts = open_buys.pop(0)["timestamp"] if open_buys else row["timestamp"]
                    raw_pnl = str(row.get("net_pnl", 0)).replace("$", "").replace(",", "")
                    try:
                        pnl = float(raw_pnl)
                    except ValueError:
                        pnl = 0.0
                    completed_trades.append({
                        "symbol": symbol,
                        "entry_timestamp": str(entry_ts),
                        "exit_timestamp": str(row.get("timestamp", "")),
                        "net_pnl": pnl,
                        "reason": str(row.get("reason", "")),
                    })
            return completed_trades
        elif "net_pnl" in df_hist.columns and ("entry_timestamp" in df_hist.columns or "timestamp" in df_hist.columns):
            ts_col = "entry_timestamp" if "entry_timestamp" in df_hist.columns else "timestamp"
            for _, row in df_hist.iterrows():
                sym = str(row.get("symbol", "")).replace("/", "-").upper()
                if not sym or sym == clean_target:
                    completed_trades.append({
                        "symbol": symbol,
                        "entry_timestamp": str(row[ts_col]),
                        "net_pnl": float(row["net_pnl"]),
                        "reason": str(row.get("reason", "")),
                    })
            return completed_trades

    # Case 3: CSV File on disk
    file_path = Path(trade_history) if trade_history else TRADE_HISTORY_FILE
    if not file_path.exists():
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            open_buys = []
            for row in reader:
                sym = str(row.get("symbol", "")).replace("/", "-").upper()
                if sym != clean_target and sym:
                    continue
                action = str(row.get("action", "")).upper()
                if action == "BUY":
                    open_buys.append(row)
                elif action.startswith("SELL"):
                    entry_ts = open_buys.pop(0)["timestamp"] if open_buys else row["timestamp"]
                    raw_pnl = str(row.get("net_pnl", "0")).replace("$", "").replace(",", "")
                    try:
                        pnl = float(raw_pnl)
                    except ValueError:
                        pnl = 0.0
                    completed_trades.append({
                        "symbol": symbol,
                        "entry_timestamp": entry_ts,
                        "exit_timestamp": row.get("timestamp", ""),
                        "net_pnl": pnl,
                        "reason": row.get("reason", ""),
                    })
    except Exception as exc:
        logger.warning("Failed to parse trade history file %s: %s", file_path, exc)

    return completed_trades


def apply_experience_replay(
    df: pd.DataFrame,
    trades: List[Dict[str, Any]],
    failure_weight: float = FAILURE_SAMPLE_WEIGHT,
    success_weight: float = SUCCESS_SAMPLE_WEIGHT,
    window_hours: float = EXPERIENCE_REPLAY_WINDOW_HOURS,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int]]:
    """
    Applies Failure Experience Replay weighting and label adjustment:
    - Default sample weight: 1.0
    - If past trade ended in LOSS (net_pnl < 0): sample_weight = 3.0, target label overridden to 0 (Failure Penalty).
    - If past trade ended in PROFIT (net_pnl > 0): sample_weight = 1.5, target label overridden to 1 (Success Boost).
    Matches past trade entry timestamp within a +/- window_hours range of df timestamps.
    """
    df_mod = df.copy()
    n_samples = len(df_mod)
    sample_weights = np.ones(n_samples, dtype=float)

    if "target" not in df_mod.columns:
        targets = np.zeros(n_samples, dtype=int)
    else:
        targets = df_mod["target"].values.copy()

    summary = {"matched": 0, "penalties": 0, "boosts": 0}
    if not trades or df_mod.empty:
        df_mod["target"] = targets
        return sample_weights, df_mod, summary

    # Standardize DataFrame index to UTC DatetimeIndex
    df_idx = df_mod.index
    if not isinstance(df_idx, pd.DatetimeIndex):
        try:
            df_idx = pd.to_datetime(df_idx)
        except Exception:
            return sample_weights, df_mod, summary

    if df_idx.tz is None:
        df_idx = df_idx.tz_localize(timezone.utc)
    else:
        df_idx = df_idx.tz_convert(timezone.utc)

    window_sec = float(window_hours) * 3600.0

    for trade in trades:
        entry_str = trade.get("entry_timestamp")
        if not entry_str:
            continue
        try:
            entry_dt = pd.to_datetime(entry_str)
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.tz_localize(timezone.utc)
            else:
                entry_dt = entry_dt.tz_convert(timezone.utc)
        except Exception:
            continue

        time_diffs = np.abs((df_idx - entry_dt).total_seconds())
        close_mask = time_diffs <= window_sec
        if not np.any(close_mask):
            continue

        summary["matched"] += 1
        matched_indices = np.where(close_mask)[0]
        pnl = float(trade.get("net_pnl", 0.0))

        if pnl < 0:
            # Loss trade: Assign 3.0 weight and enforce target = 0
            sample_weights[matched_indices] = failure_weight
            targets[matched_indices] = 0
            summary["penalties"] += len(matched_indices)
        elif pnl > 0:
            # Profit trade: Assign 1.5 weight and enforce target = 1
            sample_weights[matched_indices] = success_weight
            targets[matched_indices] = 1
            summary["boosts"] += len(matched_indices)

    df_mod["target"] = targets
    logger.info(
        "Experience Replay Applied: %d trades matched | %d bars penalized (%.1fx weight, target=0) | %d bars boosted (%.1fx weight, target=1)",
        summary["matched"], summary["penalties"], failure_weight, summary["boosts"], success_weight,
    )
    return sample_weights, df_mod, summary


def create_triple_barrier_labels(
    df: pd.DataFrame,
    atr_sl_mult: float = ATR_SL_MULTIPLIER,
    atr_tp_mult: float = ATR_TP_MULTIPLIER,
    horizon: int = HORIZON_BARS,
) -> pd.DataFrame:
    """
    Labels each bar using the Triple-Barrier Method:
    1. Upper barrier (Take Profit): Close[i] + (atr_tp_mult * ATR[i])
    2. Lower barrier (Stop Loss):   Close[i] - (atr_sl_mult * ATR[i])
    3. Vertical barrier (Time Horizon): i + horizon bars

    Labeling Logic:
    - If High touches/exceeds TP before Low touches/breaches SL -> Label 1 (Win/Buy)
    - If Low touches/breaches SL before High touches TP -> Label 0 (Loss)
    - If both touched on same candle: Conservative rule -> Label 0
    - If neither touched within horizon bars -> Label 0 (Time stop expired)

    To prevent incomplete-horizon data leakage, the last `horizon` bars are dropped
    from the returned training set.
    """
    df = df.copy()
    if "atr" not in df.columns:
        raise ValueError("DataFrame must contain 'atr' column to construct volatility barriers.")

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr"].values
    n = len(df)

    labels = np.zeros(n, dtype=int)
    barrier_events = []  # Detailed tracking: 'tp', 'sl', or 'time_expired'

    for i in range(n - horizon):
        entry_price = closes[i]
        curr_atr = atrs[i]

        tp_price = entry_price + (atr_tp_mult * curr_atr)
        sl_price = entry_price - (atr_sl_mult * curr_atr)

        outcome = 0
        event = "time_expired"

        for k in range(i + 1, i + horizon + 1):
            h_k = highs[k]
            l_k = lows[k]

            hit_tp = h_k >= tp_price
            hit_sl = l_k <= sl_price

            if hit_tp and hit_sl:
                # Ambiguous same-bar collision: Conservative assumption triggers Stop Loss
                outcome = 0
                event = "sl"
                break
            elif hit_tp:
                outcome = 1
                event = "tp"
                break
            elif hit_sl:
                outcome = 0
                event = "sl"
                break

        labels[i] = outcome
        barrier_events.append(event)

    # Pad remaining horizon rows that lacked full forward windows
    for _ in range(horizon):
        barrier_events.append("incomplete")

    df["target"] = labels
    df["barrier_event"] = barrier_events

    # Trim the trailing incomplete horizon bars for training safety
    valid_labeled_df = df.iloc[:-horizon].copy()

    win_count = (valid_labeled_df["target"] == 1).sum()
    total_count = len(valid_labeled_df)
    base_win_rate = (win_count / total_count * 100.0) if total_count > 0 else 0.0

    logger.info(
        "Triple-Barrier Labeling completed on %d bars. Base Win Rate (Label=1): %.2f%% (%d/%d)",
        total_count,
        base_win_rate,
        win_count,
        total_count,
    )
    return valid_labeled_df


class QuantModel:
    """
    Production-grade Machine Learning engine utilizing XGBoost Classifier
    specifically calibrated for financial time-series classification with strict chronological validation,
    Failure Experience Replay weighting, and Model Quality Gatekeeper.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        model_path: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.symbol = symbol or SYMBOL
        if model_path:
            self.model_path = str(model_path)
        elif symbol:
            self.model_path = str(get_model_path(symbol))
        else:
            self.model_path = str(MODEL_PATH)
        self.feature_columns = STATIONARY_FEATURE_COLUMNS
        self.model: Optional[XGBClassifier] = None

    def build_classifier(self) -> XGBClassifier:
        """
        Instantiates regularized XGBClassifier configured to prevent financial overfitting.
        """
        return XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.80,
        symbol: Optional[str] = None,
        trade_history: Optional[Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Trains the XGBoost model using a strict chronological split (default 80% train / 20% test).
        Integrates:
        1. Failure Experience Replay: Samples near past losing trades receive 3.0x weight (target 0),
           while winning trades receive 1.5x weight (target 1).
        2. Model Quality Gatekeeper: Evaluates Out-of-Sample metrics against existing model to prevent degradation.
        3. Audit Logging: Persists training metrics and decision to models/retrain_history.json.
        """
        if symbol:
            self.symbol = symbol
            if not self.model_path or "xgb_trading_model" in self.model_path:
                self.model_path = str(get_model_path(symbol, for_training=True))

        logger.info("Initiating chronological model training for %s (Train ratio: %.2f)...", self.symbol, train_ratio)

        # 1. Experience Replay Sample Weighting
        trades = extract_trade_feedback(trade_history=trade_history, symbol=self.symbol)
        sample_weights, df_replay, replay_summary = apply_experience_replay(df=df, trades=trades)

        # Prepare features and target
        X = df_replay[self.feature_columns].values
        y = df_replay["target"].values

        total_samples = len(df_replay)
        split_idx = int(total_samples * train_ratio)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        w_train = sample_weights[:split_idx].copy()

        # If any experience replay feedback samples fall in the test window,
        # replay them into X_train so the model actively learns from the experience
        replay_mask_in_test = sample_weights[split_idx:] > 1.0
        if np.any(replay_mask_in_test):
            X_replay = X_test[replay_mask_in_test]
            y_replay = y_test[replay_mask_in_test]
            w_replay = sample_weights[split_idx:][replay_mask_in_test]
            X_train = np.vstack([X_train, X_replay])
            y_train = np.concatenate([y_train, y_replay])
            w_train = np.concatenate([w_train, w_replay])
            logger.info("Experience Replay: Injected %d recent trade feedback samples into training set.", len(w_replay))

        train_dates = (df_replay.index[0], df_replay.index[split_idx - 1])
        test_dates = (df_replay.index[split_idx], df_replay.index[-1])

        logger.info(
            "Chronological Split: Train [%s to %s] (%d samples) | Test [%s to %s] (%d samples)",
            train_dates[0],
            train_dates[1],
            len(X_train),
            test_dates[0],
            test_dates[1],
            len(X_test),
        )

        # 2. Evaluate Existing / Old Model (Quality Gatekeeper Baseline)
        old_model = None
        old_metrics: Dict[str, float] = {}
        target_path = Path(self.model_path)
        if target_path.exists():
            try:
                old_data = joblib.load(target_path)
                old_model = old_data.get("model")
                if old_model is not None and hasattr(old_model, "predict_proba"):
                    old_probs = old_model.predict_proba(X_test)[:, 1]
                    old_roc = roc_auc_score(y_test, old_probs) if len(np.unique(y_test)) > 1 else 0.50
                    old_conf_mask = old_probs >= self.confidence_threshold
                    old_prec = (
                        precision_score(
                            y_test[old_conf_mask],
                            (old_probs[old_conf_mask] >= self.confidence_threshold).astype(int),
                            zero_division=0,
                        )
                        if old_conf_mask.sum() > 0
                        else 0.0
                    )
                    old_metrics = {
                        "roc_auc": float(old_roc),
                        "precision": float(old_prec),
                    }
                    logger.info("Quality Gatekeeper Baseline (Old Model): ROC-AUC=%.4f | Confident Precision=%.4f", old_roc, old_prec)
            except Exception as exc:
                logger.debug("Could not evaluate baseline model: %s", exc)

        # 3. Train New Regularized Model with Sample Weights
        self.model = self.build_classifier()
        self.model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )

        # 4. Out-of-Sample Predictions & Metrics
        test_probs = self.model.predict_proba(X_test)[:, 1]
        test_preds_default = (test_probs >= 0.50).astype(int)
        test_preds_threshold = (test_probs >= self.confidence_threshold).astype(int)

        acc = accuracy_score(y_test, test_preds_default)
        precision_default = precision_score(y_test, test_preds_default, zero_division=0)
        roc_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.50

        # High-confidence threshold metrics
        confident_mask = test_probs >= self.confidence_threshold
        num_confident_trades = int(confident_mask.sum())

        if num_confident_trades > 0:
            win_rate_threshold = float(
                (y_test[confident_mask] == 1).sum() / num_confident_trades * 100.0
            )
            precision_threshold = float(
                precision_score(
                    y_test[confident_mask], test_preds_threshold[confident_mask], zero_division=0
                )
            )
        else:
            win_rate_threshold = 0.0
            precision_threshold = 0.0

        # 5. Model Quality Gatekeeper Decision
        is_accepted = True
        gatekeeper_reason = "ACCEPTED: Initial calibration or improved performance"
        if old_model is not None and old_metrics:
            old_roc = old_metrics.get("roc_auc", 0.50)
            old_prec = old_metrics.get("precision", 0.0)
            if roc_auc < 0.50 and old_roc >= 0.50:
                is_accepted = False
                gatekeeper_reason = f"REJECTED: Sub-0.50 ROC-AUC (New: {roc_auc:.4f} vs Old: {old_roc:.4f})"
            elif (old_roc - roc_auc) > 0.05 and precision_threshold < old_prec:
                is_accepted = False
                gatekeeper_reason = f"REJECTED: Severe Metric Degradation (New ROC: {roc_auc:.4f} vs Old: {old_roc:.4f})"
            else:
                is_accepted = True
                gatekeeper_reason = f"ACCEPTED: Valid Out-of-Sample Performance (ROC: {roc_auc:.4f} vs Old: {old_roc:.4f})"

        logger.info("Quality Gatekeeper Decision: %s", gatekeeper_reason)

        if is_accepted:
            self.save_model()
        else:
            logger.warning("Quality Gatekeeper rejected model update. Restoring previous model artifact.")
            self.model = old_model

        metrics = {
            "test_accuracy": float(acc),
            "test_roc_auc": float(roc_auc),
            "default_precision": float(precision_default),
            "confident_win_rate_pct": float(win_rate_threshold),
            "confident_precision": float(precision_threshold),
            "confident_trade_signals": int(num_confident_trades),
            "test_samples": int(len(X_test)),
            "retrain_status": "ACCEPTED" if is_accepted else "REJECTED",
            "gatekeeper_reason": gatekeeper_reason,
            "trades_matched": int(replay_summary["matched"]),
            "penalties_count": int(replay_summary["penalties"]),
            "boosts_count": int(replay_summary["boosts"]),
        }

        # 6. Audit Logging to retrain_history.json
        audit_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "status": "ACCEPTED" if is_accepted else "REJECTED",
            "reason": gatekeeper_reason,
            "metrics": {
                "new_roc_auc": round(float(roc_auc), 4),
                "old_roc_auc": round(float(old_metrics.get("roc_auc", 0.0)), 4) if old_metrics else None,
                "new_precision": round(float(precision_threshold), 4),
                "old_precision": round(float(old_metrics.get("precision", 0.0)), 4) if old_metrics else None,
                "confident_trade_signals": int(num_confident_trades),
                "confident_win_rate_pct": round(float(win_rate_threshold), 2),
            },
            "experience_replay": {
                "trades_matched": int(replay_summary["matched"]),
                "penalties_3x": int(replay_summary["penalties"]),
                "boosts_1_5x": int(replay_summary["boosts"]),
            },
        }
        _append_retrain_history(audit_record)

        logger.info("=" * 60)
        logger.info("OUT-OF-SAMPLE TEST EVALUATION RESULTS (%s):", self.symbol)
        logger.info("  ROC-AUC Score          : %.4f", roc_auc)
        logger.info("  Default (P>=0.50) Prec : %.4f", precision_default)
        logger.info("  High Confidence Thresh : %.2f", self.confidence_threshold)
        logger.info("  Confident Signals Count: %d / %d", num_confident_trades, len(X_test))
        logger.info("  Confident Win Rate     : %.2f%%", win_rate_threshold)
        logger.info("  Gatekeeper Status      : %s (%s)", metrics["retrain_status"], gatekeeper_reason)
        logger.info("=" * 60)

        # Log Top Feature Importances
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_ranking = sorted(
                zip(self.feature_columns, importances), key=lambda x: x[1], reverse=True
            )
            logger.info("Top Feature Importances:")
            for feat, score in feature_ranking[:6]:
                logger.info("  %-16s: %.4f", feat, score)

        return metrics

    def save_model(self) -> None:
        """Serializes trained model to joblib disk file with trained_at timestamp."""
        if self.model is None:
            raise ValueError("No model trained yet to save.")
        now_iso = datetime.now(timezone.utc).isoformat()
        self.trained_at = now_iso
        joblib.dump(
            {
                "model": self.model,
                "features": self.feature_columns,
                "confidence_threshold": self.confidence_threshold,
                "symbol": self.symbol,
                "trained_at": now_iso,
            },
            self.model_path,
        )
        logger.info("Trained model persisted to %s (timestamp: %s)", self.model_path, now_iso)

    def load_model(self) -> None:
        """Loads serialized model from joblib file."""
        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.feature_columns = data.get("features", self.feature_columns)
        self.confidence_threshold = data.get("confidence_threshold", self.confidence_threshold)
        self.trained_at = data.get("trained_at", None)
        logger.info("Model loaded successfully from %s", self.model_path)

    def get_model_age_days(self) -> float:
        """Returns the age of the serialized model artifact in days (or inf if missing)."""
        p = Path(self.model_path)
        if not p.exists():
            return float("inf")
        if hasattr(self, "trained_at") and self.trained_at:
            try:
                dt = datetime.fromisoformat(self.trained_at.replace("Z", "+00:00"))
                return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
            except Exception:
                pass
        mtime = p.stat().st_mtime
        return (datetime.now(timezone.utc).timestamp() - mtime) / 86400.0

    def is_retrain_due(self, max_age_days: int = RETRAIN_INTERVAL_DAYS) -> bool:
        """Checks if model artifact age exceeds max_age_days (default 7 days)."""
        return self.get_model_age_days() >= max_age_days

    def predict_signal(self, latest_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Infers trading signal from the most recent bar features.
        Returns:
            Tuple of (Signal ['BUY' | 'HOLD'], Win_Probability [float])
        """
        if self.model is None:
            self.load_model()

        # Ensure we have required stationary features
        missing = set(self.feature_columns) - set(latest_df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

        latest_row = latest_df[self.feature_columns].iloc[[-1]]
        probs = self.model.predict_proba(latest_row.values)[0]
        win_prob = float(probs[1])

        signal = "BUY" if win_prob >= self.confidence_threshold else "HOLD"
        logger.info(
            "Inference: Win Probability = %.4f (Threshold: %.2f) -> Signal: %s",
            win_prob,
            self.confidence_threshold,
            signal,
        )
        return signal, win_prob
