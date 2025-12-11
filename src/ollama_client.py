# src/ollama_client.py

import json
import re
from typing import Any, Dict, List, Optional

import requests

try:
    # If you have this in your config already
    from .config import OLLAMA_MODEL_DEFAULT
except Exception:
    OLLAMA_MODEL_DEFAULT = "gemma3:4b"

OLLAMA_URL = "http://localhost:11434/api/chat"

# Reuse a single HTTP session for performance
_SESSION = requests.Session()


SYSTEM_PROMPT = """
You are a trading assistant.

You will receive structured market data for a single crypto asset at a specific time:
- Price (open, high, low, close)
- Volume
- Technical indicators (RSI, MACD, moving averages, Bollinger bands, volatility)
- Simple sentiment summary

Your task:
1. Decide how attractive it is to BUY, HOLD, or SELL this asset.
2. Output a single JSON object with this exact schema:

{
  "action_scores": {
    "Buy": 0.0-1.0,
    "Hold": 0.0-1.0,
    "Sell": 0.0-1.0
  },
  "forecast": "bullish" | "bearish" | "neutral",
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}

Rules:
- The three scores do NOT need to sum to 1.0, but must each be between 0.0 and 1.0.
- Be consistent: if forecast is "bullish", Buy usually should have a higher score than Sell.
- Return ONLY the JSON object. No markdown, no extra text.
"""


def build_user_content(row: Dict[str, Any]) -> str:
    """
    Turn a feature row into a text description for the LLM.

    This function expects keys like:
        asset, timestamp, open, high, low, close, volume,
        rsi, macd, macd_signal,
        sma_7, sma_30,
        bb_high, bb_low,
        returns_1h, volatility_24h,
        sentiment_summary (optional).
    """
    asset = row.get("asset", "UNKNOWN")
    ts = row.get("timestamp", "UNKNOWN")

    open_price = row.get("open", None)
    high = row.get("high", None)
    low = row.get("low", None)
    close = row.get("close", None)
    volume = row.get("volume", None)

    rsi = row.get("rsi", None)
    macd = row.get("macd", None)
    macd_signal = row.get("macd_signal", None)
    sma7 = row.get("sma_7", None)
    sma30 = row.get("sma_30", None)
    bb_high = row.get("bb_high", None)
    bb_low = row.get("bb_low", None)
    ret_1h = row.get("returns_1h", None)
    vol_24h = row.get("volatility_24h", None)

    sentiment = row.get("sentiment_summary", "No sentiment data available.")

    lines = [
        f"Asset: {asset}",
        f"Timestamp: {ts}",
        "",
        "Price & Volume:",
        f"- Open: {open_price}",
        f"- High: {high}",
        f"- Low: {low}",
        f"- Close: {close}",
        f"- Volume: {volume}",
        "",
        "Technical indicators:",
        f"- RSI: {rsi}",
        f"- MACD: {macd}",
        f"- MACD Signal: {macd_signal}",
        f"- SMA 7: {sma7}",
        f"- SMA 30: {sma30}",
        f"- Bollinger High: {bb_high}",
        f"- Bollinger Low: {bb_low}",
        f"- 1h Returns: {ret_1h}",
        f"- 24h Volatility (std of returns): {vol_24h}",
        "",
        "Sentiment summary:",
        f"{sentiment}",
        "",
        "Based on this information, estimate how attractive it is to BUY, HOLD, or SELL this asset now.",
        "Return ONLY the JSON object with fields action_scores, forecast, confidence, reason.",
    ]
    return "\n".join(lines)


# -----------------------
# Robust JSON extraction
# -----------------------


def _try_json_direct(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _try_json_braces(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to find a top-level {...} block by matching braces.
    """
    start = -1
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    # continue searching in case there is another valid block
                    start = -1
    return None


def _try_json_regex(text: str) -> Optional[Dict[str, Any]]:
    """
    Fallback: use regex to find the first {...} substring and parse it.
    """
    pattern = re.compile(r"\{.*\}", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _fallback_from_text(text: str) -> Dict[str, Any]:
    """
    Last-resort extraction if JSON parsing fails completely.

    We try to guess Buy/Hold/Sell scores and forecast/confidence
    from any numbers and keywords in the text.
    """
    text_lower = text.lower()

    def find_score(keyword: str) -> float:
        # Look for keyword followed by a number like 0.7 or 70%
        pattern = re.compile(rf"{keyword}[^0-9]*([0-9]+(\.[0-9]+)?)", re.IGNORECASE)
        m = pattern.search(text)
        if not m:
            return 0.0
        try:
            val = float(m.group(1))
            if val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0

    buy_score = find_score("buy")
    hold_score = find_score("hold")
    sell_score = find_score("sell")

    if buy_score == hold_score == sell_score == 0.0:
        # Fallback neutral
        buy_score = hold_score = sell_score = 1.0 / 3.0

    if "bullish" in text_lower:
        forecast = "bullish"
    elif "bearish" in text_lower:
        forecast = "bearish"
    else:
        forecast = "neutral"

    confidence = 0.5

    return {
        "action_scores": {
            "Buy": float(buy_score),
            "Hold": float(hold_score),
            "Sell": float(sell_score),
        },
        "forecast": forecast,
        "confidence": confidence,
        "reason": "Fallback extraction from non-JSON response.",
    }


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Try multiple strategies to pull a JSON object out of LLM output.
    """
    for fn in (_try_json_direct, _try_json_braces, _try_json_regex):
        obj = fn(text)
        if obj is not None:
            return obj
    # If everything failed, do heuristic fallback
    return _fallback_from_text(text)


# --------------
# Main API call
# --------------


def ask_ollama(
    model_name: str,
    row: Dict[str, Any],
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Call Ollama with given model and one feature row.

    Returns a dict with keys:
        - action_scores: {"Buy": float, "Hold": float, "Sell": float}
        - forecast: str
        - confidence: float
        - reason: str
    """
    user_content = build_user_content(row)

    payload = {
        "model": model_name or OLLAMA_MODEL_DEFAULT,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    resp = _SESSION.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()
    # Ollama's chat API returns {"message": {"content": "..."}, ...}
    content = ""
    if isinstance(data, dict):
        msg = data.get("message") or data.get("choices", [{}])[0]
        if isinstance(msg, dict):
            content = msg.get("content", "") or msg.get("text", "")
    if not content:
        raise ValueError("Empty response from Ollama")

    parsed = _extract_json(content)

    # Ensure structure
    scores = parsed.get("action_scores", {})
    for k in ("Buy", "Hold", "Sell"):
        scores.setdefault(k, 0.0)

    parsed["action_scores"] = {
        "Buy": float(scores["Buy"]),
        "Hold": float(scores["Hold"]),
        "Sell": float(scores["Sell"]),
    }
    parsed.setdefault("forecast", "neutral")
    parsed.setdefault("confidence", 0.5)
    parsed.setdefault("reason", "")

    return parsed
