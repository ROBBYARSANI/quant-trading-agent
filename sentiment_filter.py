"""
Sentiment Filter Module for Autonomous Quantitative AI Trading Agent.
Implements a 100% Free, Standalone News Sentiment Gatekeeper using Yahoo Finance News
and Google News RSS feeds with Lazy-Loaded FinBERT (ProsusAI/finbert) and Financial Lexicon fallback.

Veto & Boost Rules:
- If composite_sentiment_score < -0.15 -> VETO (Cancel BUY signal due to market FUD)
- If composite_sentiment_score > +0.25 -> BOOST (Add +0.03 to win_prob, capped at MAX_RISK_PER_TRADE)
"""

from datetime import datetime, timezone
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

from config import (
    SENTIMENT_BOOST_THRESHOLD,
    SENTIMENT_BOOST_VALUE,
    SENTIMENT_VETO_THRESHOLD,
    SYMBOL,
    setup_logger,
)
from data_pipeline import to_yfinance_symbol

logger = setup_logger("SentimentFilter")

# Curated Financial Lexicon for offline and lightweight sentiment scoring
BULLISH_LEXICON = {
    "bull", "bullish", "surge", "surged", "rally", "rallies", "gain", "gains",
    "gained", "breakout", "jump", "jumped", "profit", "profitable", "high",
    "highs", "upside", "record", "accumulate", "accumulation", "adoption",
    "adopted", "recovery", "recover", "partner", "partnership", "green",
    "approve", "approved", "approval", "inflow", "inflows", "beat", "pump",
    "soar", "soared", "peak", "optimistic", "growth", "boost", "support",
    "breakthrough", "milestone", "institutional", "all-time", "ath", "rebound"
}

BEARISH_LEXICON = {
    "bear", "bearish", "drop", "dropped", "drops", "crash", "crashed", "crashes",
    "slump", "slumped", "fall", "fell", "fallen", "loss", "losses", "plunge",
    "plunged", "risk", "risks", "decline", "declined", "low", "lows", "ban",
    "banned", "hack", "hacked", "fraud", "scam", "dump", "dumped", "fear",
    "panic", "probe", "probed", "investigation", "liquidate", "liquidation",
    "outflow", "outflows", "red", "lawsuit", "sued", "sec", "sanction",
    "sanctions", "bleed", "bleeding", "decay", "warning", "threat", "pessimistic"
}


class CryptoSentimentFilter:
    """
    Evaluates market sentiment using public, unblocked news feeds (Yahoo Finance + Google News RSS).
    Utilizes Lazy Loading for ProsusAI/finbert with automatic Financial Lexicon fallback.
    Requires ZERO API keys and operates seamlessly in any network environment.
    """

    def __init__(
        self,
        veto_threshold: float = SENTIMENT_VETO_THRESHOLD,
        boost_threshold: float = SENTIMENT_BOOST_THRESHOLD,
        boost_value: float = SENTIMENT_BOOST_VALUE,
    ):
        self.veto_threshold = veto_threshold
        self.boost_threshold = boost_threshold
        self.boost_value = boost_value
        self._finbert_pipeline = None

    def _load_finbert_pipeline(self) -> Any:
        """
        Lazy-loads ProsusAI/finbert sentiment pipeline into memory only when invoked.
        Falls back to financial lexicon if model download fails or dependencies are missing.
        """
        if self._finbert_pipeline is not None:
            return self._finbert_pipeline

        try:
            from transformers import pipeline
            logger.info("Lazy-loading 'ProsusAI/finbert' NLP sentiment pipeline...")
            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
            )
            logger.info("ProsusAI/finbert pipeline loaded successfully into memory.")
        except Exception as exc:
            logger.warning(
                "Could not initialize ProsusAI/finbert (%s). Falling back to Financial Lexicon NLP engine.",
                exc,
            )
            self._finbert_pipeline = "LEXICON_FALLBACK"

        return self._finbert_pipeline

    def fetch_yahoo_news(self, symbol: str = SYMBOL, limit: int = 10) -> List[str]:
        """Fetches news from Yahoo Finance ticker object."""
        yf_sym = to_yfinance_symbol(symbol)
        headlines = []
        try:
            ticker = yf.Ticker(yf_sym)
            raw_news = ticker.news
            if raw_news and isinstance(raw_news, list):
                for item in raw_news[:limit]:
                    text = ""
                    if isinstance(item, dict):
                        if "content" in item and isinstance(item["content"], dict):
                            title = item["content"].get("title", "")
                            summary = item["content"].get("summary", "")
                            text = f"{title}. {summary}".strip()
                        else:
                            title = item.get("title", "") or item.get("headline", "")
                            summary = item.get("summary", "")
                            text = f"{title}. {summary}".strip()
                    if text and text not in headlines:
                        headlines.append(text)
        except Exception as exc:
            logger.debug("Yahoo Finance news fetch error for %s: %s", yf_sym, exc)
        return headlines

    def fetch_rss_news(self, query: str = "Bitcoin crypto market", limit: int = 10) -> List[str]:
        """Fetches live public news headlines via Google News RSS (zero API keys)."""
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        headlines = []
        try:
            req = urllib.request.Request(
                rss_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            items = root.findall("./channel/item")
            for it in items[:limit]:
                title_elem = it.find("title")
                if title_elem is not None and title_elem.text:
                    clean_title = title_elem.text.strip()
                    # Strip trailing publisher attribution e.g. " - CoinDesk"
                    if " - " in clean_title:
                        clean_title = clean_title.rsplit(" - ", 1)[0]
                    if clean_title and clean_title not in headlines:
                        headlines.append(clean_title)

        except Exception as exc:
            logger.debug("Google News RSS fetch error: %s", exc)

        return headlines

    def fetch_recent_news(
        self, symbol: str = SYMBOL, limit: int = 10
    ) -> List[str]:
        """
        Combines news from Yahoo Finance and Google News RSS to guarantee fresh headlines.
        """
        combined = []
        # 1. Try Yahoo Finance news
        yf_news = self.fetch_yahoo_news(symbol=symbol, limit=limit)
        combined.extend(yf_news)

        # 2. Complement with Google News RSS
        query = "Bitcoin crypto market" if "BTC" in symbol.upper() else f"{symbol} crypto market"
        rss_news = self.fetch_rss_news(query=query, limit=limit)
        for h in rss_news:
            if h not in combined and len(combined) < limit:
                combined.append(h)

        logger.info("Retrieved %d real-time news headlines for %s.", len(combined), symbol)
        return combined[:limit]

    def score_text_lexicon(self, text: str) -> float:
        """Lexicon-based scoring engine returning [-1.0, 1.0]."""
        words = set(re.findall(r"\b[a-z\-]+\b", text.lower()))
        bull_matches = len(words.intersection(BULLISH_LEXICON))
        bear_matches = len(words.intersection(BEARISH_LEXICON))

        total = bull_matches + bear_matches
        if total == 0:
            return 0.0

        return (bull_matches - bear_matches) / total

    def analyze_sentiment(self, news_items: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Analyzes news headlines and calculates a composite score between -1.0 (bearish/FUD) and +1.0 (bullish).
        """
        if not news_items:
            logger.info("No news items retrieved. Neutral score (0.0) assigned.")
            return 0.0, []

        pipe = self._load_finbert_pipeline()
        item_scores: List[float] = []
        breakdown: List[Dict[str, Any]] = []

        for text in news_items:
            clean_text = text[:512]

            if pipe != "LEXICON_FALLBACK" and callable(pipe):
                try:
                    res = pipe(clean_text)[0]
                    lbl = res["label"].lower()
                    conf = float(res["score"])

                    score = conf if lbl == "positive" else (-conf if lbl == "negative" else 0.0)
                    item_scores.append(score)
                    breakdown.append({
                        "headline": clean_text[:80] + "...",
                        "engine": "finbert",
                        "label": lbl,
                        "score": score,
                    })
                    continue
                except Exception as exc:
                    logger.debug("FinBERT inference failed on text (%s), using lexicon.", exc)

            # Lexicon Fallback
            score = self.score_text_lexicon(clean_text)
            lbl = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
            item_scores.append(score)
            breakdown.append({
                "headline": clean_text[:80] + "...",
                "engine": "lexicon",
                "label": lbl,
                "score": score,
            })

        composite_score = float(sum(item_scores) / len(item_scores)) if item_scores else 0.0
        composite_score = max(-1.0, min(1.0, composite_score))

        logger.info(
            "Sentiment Analysis Complete: Analyzed %d articles | Composite Score = %+.4f",
            len(news_items),
            composite_score,
        )
        return composite_score, breakdown

    def evaluate_sentiment_gate(
        self, symbol: str, base_win_prob: float, sample_news: Optional[List[str]] = None
    ) -> Tuple[bool, float, float, str]:
        """
        Master Gatekeeper Decision:
        - composite_score < -0.15 -> VETO (Cancel BUY signal)
        - composite_score > +0.25 -> BOOST (win_prob + 0.03)
        - otherwise               -> APPROVED (Unmodified win_prob)
        """
        news = sample_news if sample_news is not None else self.fetch_recent_news(symbol=symbol)
        score, _ = self.analyze_sentiment(news)

        if score < self.veto_threshold:
            reason = (
                f"VETOED: Composite sentiment score ({score:+.3f}) is below "
                f"veto threshold ({self.veto_threshold:+.2f}). Signal canceled due to FUD/negative news."
            )
            logger.warning(reason)
            return False, base_win_prob, score, reason

        if score > self.boost_threshold:
            boosted_prob = min(base_win_prob + self.boost_value, 0.99)
            reason = (
                f"BOOSTED: Composite sentiment score ({score:+.3f}) exceeds "
                f"boost threshold ({self.boost_threshold:+.2f}). Win probability boosted from "
                f"{base_win_prob:.4f} to {boosted_prob:.4f} (+{self.boost_value:.2f})."
            )
            logger.info(reason)
            return True, boosted_prob, score, reason

        reason = (
            f"APPROVED: Composite sentiment score ({score:+.3f}) is neutral "
            f"[{self.veto_threshold:+.2f}, {self.boost_threshold:+.2f}]. Unmodified win_prob ({base_win_prob:.4f})."
        )
        logger.info(reason)
        return True, base_win_prob, score, reason
