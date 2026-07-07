import requests
import pandas as pd
import time

INTERVAL_MAP = {
    "1m": "Min1", "5m": "Min5", "15m": "Min15", "30m": "Min30",
    "1h": "Min60", "4h": "Hour4", "8h": "Hour8",
    "1d": "Day1", "1w": "Week1", "1M": "Month1"
}

def fetch_klines(symbol: str, interval: str = "4h", limit: int = 300) -> pd.DataFrame:
    """Fetch OHLCV from MEXC Futures. Returns DataFrame with columns: open_time, open, high, low, close, volume"""
    mexc_interval = INTERVAL_MAP.get(interval, interval)
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}_USDT"

    end = int(time.time())
    interval_seconds = {
        "Min1": 60, "Min5": 300, "Min15": 900, "Min30": 1800,
        "Min60": 3600, "Hour4": 14400, "Hour8": 28800,
        "Day1": 86400, "Week1": 604800, "Month1": 2592000
    }
    sec = interval_seconds.get(mexc_interval, 14400)
    start = end - (limit * sec)

    resp = requests.get(url, params={"interval": mexc_interval, "start": start, "end": end})
    resp.raise_for_status()
    data = resp.json()["data"]

    df = pd.DataFrame({
        "open_time": [t * 1000 for t in data["time"]],
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data["vol"]
    })
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df
