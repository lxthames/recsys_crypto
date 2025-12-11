import ccxt
import pandas as pd
from pathlib import Path
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from .config import ASSETS, EXCHANGE, TIMEFRAME, WINDOW_DAYS, DATA_DIR


def get_exchange():
    return getattr(ccxt, EXCHANGE)()


def fetch_ohlcv_for_symbol(exchange, symbol: str, limit: int | None = None) -> pd.DataFrame:
    if limit is None:
        limit = WINDOW_DAYS * 24  # rough for 1h data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["sma_7"] = SMAIndicator(df["close"], window=7).sma_indicator()
    df["sma_30"] = SMAIndicator(df["close"], window=30).sma_indicator()

    bb = BollingerBands(df["close"], window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df["returns_1h"] = df["close"].pct_change()
    df["volatility_24h"] = df["returns_1h"].rolling(24).std()

    df = df.dropna()
    return df


def build_market_feature_table() -> pd.DataFrame:
    data_path = Path(DATA_DIR) / "processed"
    data_path.mkdir(parents=True, exist_ok=True)

    exchange = get_exchange()
    all_frames = []

    for symbol in ASSETS:
        symbol_with_slash = symbol.replace("USDT", "/USDT")
        df = fetch_ohlcv_for_symbol(exchange, symbol_with_slash)
        df = add_technical_indicators(df)
        df["asset"] = symbol
        all_frames.append(df)

    full = pd.concat(all_frames, ignore_index=True)
    out_file = data_path / "market_features.csv"
    full.to_csv(out_file, index=False)
    print(f"Saved market features to {out_file}")
    return full


if __name__ == "__main__":
    build_market_feature_table()
