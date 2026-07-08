#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_mexc import fetch_klines
from channels import detect_channel
from chart import draw_channel_chart

symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "BTC"
interval = sys.argv[2] if len(sys.argv) > 2 else "4h"
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 200

print(f"Fetching {symbol}USDT {interval} ({limit} candles)...")
df = fetch_klines(symbol, interval, limit)
print(f"Got {len(df)} candles")

channel = detect_channel(df)
path = draw_channel_chart(symbol, df, channel, interval)
print(f"Chart saved: {path}")

if channel:
    print(f"Direction: {channel['direction']}")
    print(f"Width: {channel['width_pct']:.2f}%")
    print(f"Position: {channel['price_position']:.1f}%")
    print(f"Touches: upper={channel['touches_upper']}, lower={channel['touches_lower']}")
    if channel.get('breakout'):
        print(f"BREAKOUT: {channel['breakout']}")
    if channel.get('predicted_channel'):
        print("Predicted channel drawn")
else:
    print("No channel detected")
