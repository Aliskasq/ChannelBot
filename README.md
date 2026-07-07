# Channel Bot 📐

Automatic price channel detection for crypto futures (MEXC).

## Features
- **Swing-based channels** — finds channels through swing highs/lows (body max/min, not wicks)
- **Log-space math** — lines are straight in log scale (correct for crypto)
- **Channel prediction** — predicts next channel on breakout (same width or ×0.618 fibo)
- **Chart generation** — candlestick charts with channel overlay, RSI panel
- **MEXC Futures** — fetches candles from MEXC contract API (no API key needed)

## Usage

```bash
pip install mplfinance matplotlib numpy pandas requests scipy pillow

python run.py BTC 4h        # Bitcoin 4-hour
python run.py ETH 1h        # Ethereum 1-hour
python run.py RIVER 15m     # Any MEXC futures pair
python run.py SOL 1d 500    # 500 candles
```

## Supported intervals
`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `8h`, `1d`, `1w`, `1M`

## How it works

1. **Fetch candles** from MEXC Futures API (no auth required)
2. **Find swing points** — local maxima/minima of candle bodies with adaptive radius
3. **Build channel candidates** — fit lines through pairs of swing highs/lows in log space
4. **Validate** — min 2 touches per side, max 20% breaches, min 30 candles span
5. **Score & select** — prefer more touches, wider span, price inside channel
6. **Predict breakout** — if price breaks channel, project new channel

## Output

- PNG chart with candlesticks, channel lines (cyan), predicted channel (yellow dashed), swing points, RSI panel
- Channel info: direction, width %, price position, touches, breakout status

## Files
- `fetch_mexc.py` — MEXC Futures kline fetcher
- `channels.py` — channel detection algorithm
- `chart.py` — chart drawing (mplfinance)
- `run.py` — CLI entry point
