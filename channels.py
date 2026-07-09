"""
Channel detection algorithm v6.
Alisa's algorithm:
1. From current candle, scan back 200 candles. Find the HIGHEST peak.
2. Find the next significant peak to the RIGHT of it (lower) → upper line of CH1.
3. Lower line = SAME SLOPE, shifted to the lowest candle in the channel span.
4. Break = candle fully exits below/above the lower/upper line → trend change.
5. CH2 starts from CH1's last anchor (shared point), next peak → new channel.
6. CH2 lower/upper = same slope, placed on last bounce before break.
"""
import numpy as np
import pandas as pd


def _smart_high(df):
    """If upper wick > 2x body → use body top, else use high."""
    high = df["high"].values.astype(float)
    opn = df["open"].values.astype(float)
    close = df["close"].values.astype(float)
    body_top = np.maximum(opn, close)
    body_size = np.abs(close - opn)
    upper_wick = high - body_top
    body_safe = np.where(body_size == 0, 1e10, body_size)
    use_body = upper_wick > 2.0 * body_safe
    return np.where(use_body, body_top, high)


def _smart_low(df):
    """If lower wick > 2x body → use body bottom, else use low."""
    low = df["low"].values.astype(float)
    opn = df["open"].values.astype(float)
    close = df["close"].values.astype(float)
    body_bot = np.minimum(opn, close)
    body_size = np.abs(close - opn)
    lower_wick = body_bot - low
    body_safe = np.where(body_size == 0, 1e10, body_size)
    use_body = lower_wick > 2.0 * body_safe
    return np.where(use_body, body_bot, low)


def find_swing_highs(df, min_radius=2, max_radius=6, min_distance=3):
    """Find swing high points using smart_high (body-based if wick too long)."""
    prices = _smart_high(df)
    n = len(prices)
    candidates = []
    for i in range(min_radius, n - min_radius):
        for r in range(min_radius, min(max_radius + 1, min(i + 1, n - i))):
            left = prices[max(0, i - r):i]
            right = prices[i + 1:min(n, i + r + 1)]
            if len(left) > 0 and len(right) > 0 and prices[i] >= max(np.max(left), np.max(right)):
                candidates.append((i, float(prices[i])))
                break
    if not candidates:
        return []
    candidates.sort(key=lambda x: -x[1])
    result = []
    for idx, price in candidates:
        if all(abs(idx - r[0]) >= min_distance for r in result):
            result.append((idx, price))
    result.sort(key=lambda x: x[0])
    return result


def find_swing_lows(df, min_radius=2, max_radius=6, min_distance=3):
    """Find swing low points using smart_low."""
    prices = _smart_low(df)
    n = len(prices)
    candidates = []
    for i in range(min_radius, n - min_radius):
        for r in range(min_radius, min(max_radius + 1, min(i + 1, n - i))):
            left = prices[max(0, i - r):i]
            right = prices[i + 1:min(n, i + r + 1)]
            if len(left) > 0 and len(right) > 0 and prices[i] <= min(np.min(left), np.min(right)):
                candidates.append((i, float(prices[i])))
                break
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[1])
    result = []
    for idx, price in candidates:
        if all(abs(idx - r[0]) >= min_distance for r in result):
            result.append((idx, price))
    result.sort(key=lambda x: x[0])
    return result


def _log_line(ia, pa, ib, pb):
    """Log-space line through two price points. Returns (slope, intercept)."""
    if ib == ia:
        return 0.0, np.log(pa)
    s = (np.log(pb) - np.log(pa)) / (ib - ia)
    return s, np.log(pa) - s * ia


def _price_at(s, b, i):
    """Price at index i for log-space line (slope, intercept)."""
    return np.exp(s * i + b)


def _touches(points, s, b, tol=0.015):
    """Find points that touch a log-space line within tolerance."""
    return [(i, p) for i, p in points
            if abs(p - _price_at(s, b, i)) / _price_at(s, b, i) <= tol]


def _direction(s):
    if abs(s) < 0.00005:
        return "horizontal"
    return "ascending" if s > 0 else "descending"


def detect_channel(df):
    """
    Main channel detection — Alisa's algorithm.
    
    1. Find highest peak in last 200 candles
    2. Find next significant peak to the RIGHT (lower) → upper line
    3. Lower line = same slope on lowest point
    4. Break detection → CH2 with shared anchor
    """
    n = len(df)
    if n < 50:
        return None

    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)
    smart_lows = _smart_low(df)
    smart_highs = _smart_high(df)
    # Body bottoms for lower line placement (not wicks!)
    body_bottoms = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))

    if len(swing_highs) < 2:
        return None

    # =============================================
    # STEP 1: Find the HIGHEST peak (anchor 1)
    # =============================================
    # Only look at the last 200 candles
    lookback_start = max(0, n - 200)
    relevant_highs = [(i, p) for i, p in swing_highs if i >= lookback_start]
    
    if len(relevant_highs) < 2:
        return None
    
    # Anchor 1 = the absolute highest swing high
    anchor1 = max(relevant_highs, key=lambda x: x[1])
    
    # =============================================
    # STEP 2: Find anchor 2 — next peak to the RIGHT, lower
    # =============================================
    # Candidates: swing highs to the RIGHT of anchor1, that are LOWER
    # Minimum 30 candles apart for meaningful channel slope
    right_peaks = [(i, p) for i, p in relevant_highs 
                   if i > anchor1[0] and p < anchor1[1] and (i - anchor1[0]) >= 30]
    
    if not right_peaks:
        # No peaks to the right — try looking for ascending channel instead
        # (find lowest valley, then next valley to the right that's higher)
        return _try_ascending_channel(df, swing_highs, swing_lows, smart_highs, smart_lows, n)
    
    # Pick the most prominent (highest) peak to the right
    # This gives the best channel definition
    anchor2 = max(right_peaks, key=lambda x: x[1])
    
    # =============================================
    # STEP 3: Upper line through anchor1 → anchor2
    # =============================================
    slope, upper_int = _log_line(anchor1[0], anchor1[1], anchor2[0], anchor2[1])
    
    # Sanity: slope shouldn't be too steep
    if abs(slope) > 0.03:
        return None
    
    upper_touches = _touches(swing_highs, slope, upper_int, 0.015)
    
    # =============================================
    # STEP 3b: If 4+ touches AND channel too wide, REBUILD from 2 closest to price
    # =============================================
    # Quick width check before rebuild
    _mid = (anchor1[0] + anchor2[0]) // 2
    _span_s = anchor1[0]
    _span_e = min(n, anchor2[0] + 10)
    _rl = body_bottoms[_span_s:_span_e]
    _quick_width = 0
    if len(_rl) > 0:
        _li = np.argmin(_rl)
        _lp = float(_rl[_li])
        _lint = np.log(_lp) - slope * (_span_s + _li)
        _um = _price_at(slope, upper_int, _mid)
        _lm = _price_at(slope, _lint, _mid)
        if _lm > 0 and _um > _lm:
            _quick_width = (_um - _lm) / _lm * 100
    
    if len(upper_touches) >= 4 and _quick_width > 20:
        # Take the 2 rightmost (closest to current price) touch points
        sorted_by_idx = sorted(upper_touches, key=lambda x: x[0], reverse=True)
        new_a2 = sorted_by_idx[0]  # closest to price
        new_a1 = sorted_by_idx[1]  # second closest
        # Make sure a1 is left of a2
        if new_a1[0] > new_a2[0]:
            new_a1, new_a2 = new_a2, new_a1
        anchor1 = new_a1
        anchor2 = new_a2
        slope, upper_int = _log_line(anchor1[0], anchor1[1], anchor2[0], anchor2[1])
        if abs(slope) > 0.03:
            return None
        upper_touches = _touches(swing_highs, slope, upper_int, 0.015)
    
    # =============================================
    # STEP 4: Lower line — SAME SLOPE, on lowest body bottom
    # =============================================
    # Find the lowest BODY BOTTOM BETWEEN the two anchor points (not wicks!)
    span_start = anchor1[0]
    span_end = min(n, anchor2[0] + 10)
    
    region_lows = body_bottoms[span_start:span_end]
    if len(region_lows) == 0:
        return None
    
    min_rel_idx = np.argmin(region_lows)
    low_idx = span_start + min_rel_idx
    low_price = float(region_lows[min_rel_idx])
    
    lower_int = np.log(low_price) - slope * low_idx
    
    # Validate channel width
    mid = (anchor1[0] + anchor2[0]) // 2
    um = _price_at(slope, upper_int, mid)
    lm = _price_at(slope, lower_int, mid)
    if lm >= um:
        return None
    width_pct = (um - lm) / lm * 100
    if width_pct < 1.0:
        return None
    
    lower_touches = _touches(swing_lows, slope, lower_int, 0.015)
    
    # =============================================
    # STEP 5: Find break point
    # =============================================
    # Break DOWN: candle fully below lower line (high < lower line)
    break_idx = None
    breakout = None
    
    search_start = low_idx  # start looking from the lowest point
    for i in range(search_start, n):
        lower_at = _price_at(slope, lower_int, i)
        high_i = float(df["high"].iloc[i])
        # Candle fully below lower line
        if high_i < lower_at * 0.997:
            break_idx = i
            breakout = "down"
            break
    
    # If no break down, check break UP
    if break_idx is None:
        for i in range(anchor2[0], n):
            upper_at = _price_at(slope, upper_int, i)
            low_i = float(df["low"].iloc[i])
            if low_i > upper_at * 1.003:
                break_idx = i
                breakout = "up"
                break
    
    # =============================================
    # STEP 6: CH2 — new channel after break
    # =============================================
    second_channel = None
    
    if break_idx is not None and breakout == "down":
        # CH2 upper line: anchor1 = CH1's anchor2 (SHARED POINT!)
        ch2_a1 = anchor2  # shared point
        
        # Find next significant peak AFTER break
        post_peaks = [(i, p) for i, p in swing_highs 
                      if i > ch2_a1[0] and (i - ch2_a1[0]) >= 5]
        
        if post_peaks:
            # Pick the LATEST (most recent) peak — shows current trend direction
            ch2_a2 = max(post_peaks, key=lambda x: x[0])
            
            ch2_slope, ch2_upper_int = _log_line(ch2_a1[0], ch2_a1[1], ch2_a2[0], ch2_a2[1])
            
            if abs(ch2_slope) <= 0.03:
                ch2_upper_touches = _touches(swing_highs, ch2_slope, ch2_upper_int, 0.015)
                
                # CH2 lower: same slope, on LAST REAL BOUNCE in CH1
                # A "bounce" = a CH1 lower-line touch where price actually went
                # back UP into the channel (not just touched and kept falling)
                ch1_lower_touches = _touches(swing_lows, slope, lower_int, 0.02)
                
                ch2_low_anchor = None
                # Walk CH1 lower touches in reverse, find last real bounce
                for ti, tp in reversed(ch1_lower_touches):
                    if ti >= break_idx:
                        continue  # skip touches in/after break zone
                    # Check if price bounced: within next 15 candles, 
                    # did a high go above lower_line + 30% of channel width?
                    ch_width_at = _price_at(slope, upper_int, ti) - _price_at(slope, lower_int, ti)
                    bounce_threshold = _price_at(slope, lower_int, ti) + ch_width_at * 0.3
                    bounced = False
                    for j in range(ti + 1, min(n, ti + 15)):
                        if float(df["high"].iloc[j]) > bounce_threshold:
                            bounced = True
                            break
                    if bounced:
                        ch2_low_anchor = (ti, tp)
                        break
                else:
                    # Fallback: lowest smart_low in range
                    region = smart_lows[max(0, ch2_a1[0]):min(n, break_idx + 5)]
                    if len(region) > 0:
                        ri = np.argmin(region)
                        ch2_low_anchor = (ch2_a1[0] + ri, float(region[ri]))
                    else:
                        ch2_low_anchor = None
                
                if ch2_low_anchor:
                    ch2_lower_int = np.log(ch2_low_anchor[1]) - ch2_slope * ch2_low_anchor[0]
                    
                    mid2 = (ch2_a1[0] + ch2_a2[0]) // 2
                    um2 = _price_at(ch2_slope, ch2_upper_int, mid2)
                    lm2 = _price_at(ch2_slope, ch2_lower_int, mid2)
                    
                    if um2 > lm2:
                        ch2_width = (um2 - lm2) / lm2 * 100
                        ch2_lower_touches = _touches(swing_lows, ch2_slope, ch2_lower_int, 0.015)
                        
                        last_idx = n - 1
                        last_close = float(df["close"].iloc[-1])
                        u2_now = _price_at(ch2_slope, ch2_upper_int, last_idx)
                        l2_now = _price_at(ch2_slope, ch2_lower_int, last_idx)
                        pos2 = (last_close - l2_now) / (u2_now - l2_now) * 100 if u2_now > l2_now else 50
                        
                        second_channel = {
                            "direction": _direction(ch2_slope),
                            "upper_line": {"slope": ch2_slope, "intercept": ch2_upper_int,
                                           "points": ch2_upper_touches},
                            "lower_line": {"slope": ch2_slope, "intercept": ch2_lower_int,
                                           "points": ch2_lower_touches},
                            "width_pct": ch2_width,
                            "price_position": pos2,
                            "touches_upper": len(ch2_upper_touches),
                            "touches_lower": len(ch2_lower_touches),
                            "anchors": {
                                "upper": [ch2_a1, ch2_a2],
                                "lower": [ch2_low_anchor],
                            },
                        }
    
    # Current position relative to CH1
    last_idx = n - 1
    last_close = float(df["close"].iloc[-1])
    upper_now = _price_at(slope, upper_int, last_idx)
    lower_now = _price_at(slope, lower_int, last_idx)
    position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0
    
    return {
        "direction": _direction(slope),
        "upper_line": {"slope": slope, "intercept": upper_int,
                       "points": upper_touches},
        "lower_line": {"slope": slope, "intercept": lower_int,
                       "points": lower_touches},
        "width_pct": width_pct,
        "price_position": position,
        "breakout": breakout,
        "second_channel": second_channel,
        "predicted_channel": None,
        "touches_upper": len(upper_touches),
        "touches_lower": len(lower_touches),
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "anchors": {
            "upper": [anchor1, anchor2],
            "lower": [(low_idx, low_price)],
        },
    }


def _try_ascending_channel(df, swing_highs, swing_lows, smart_highs, smart_lows, n):
    """
    Mirror algorithm for ascending channels:
    1. Find lowest valley in last 200 candles
    2. Find next valley to the RIGHT (higher) → lower line
    3. Upper line = same slope on highest point
    """
    lookback_start = max(0, n - 200)
    relevant_lows = [(i, p) for i, p in swing_lows if i >= lookback_start]
    
    if len(relevant_lows) < 2:
        return None
    
    # Anchor 1 = absolute lowest valley
    anchor1 = min(relevant_lows, key=lambda x: x[1])
    
    # Anchor 2 = next valley to the RIGHT, higher
    right_valleys = [(i, p) for i, p in relevant_lows
                     if i > anchor1[0] and p > anchor1[1] and (i - anchor1[0]) >= 5]
    
    if not right_valleys:
        return None
    
    anchor2 = min(right_valleys, key=lambda x: x[1])
    
    slope, lower_int = _log_line(anchor1[0], anchor1[1], anchor2[0], anchor2[1])
    
    if abs(slope) > 0.03:
        return None
    
    lower_touches = _touches(swing_lows, slope, lower_int, 0.015)
    
    # Upper line = same slope, on highest point in span
    span_start = anchor1[0]
    span_end = min(n, anchor2[0] + 20)
    region_highs = smart_highs[span_start:span_end]
    if len(region_highs) == 0:
        return None
    
    max_rel_idx = np.argmax(region_highs)
    high_idx = span_start + max_rel_idx
    high_price = float(region_highs[max_rel_idx])
    
    upper_int = np.log(high_price) - slope * high_idx
    
    mid = (anchor1[0] + anchor2[0]) // 2
    um = _price_at(slope, upper_int, mid)
    lm = _price_at(slope, lower_int, mid)
    if lm >= um:
        return None
    width_pct = (um - lm) / lm * 100
    if width_pct < 1.0:
        return None
    
    upper_touches = _touches(swing_highs, slope, upper_int, 0.015)
    
    # Break UP
    breakout = None
    for i in range(anchor2[0], n):
        upper_at = _price_at(slope, upper_int, i)
        low_i = float(df["low"].iloc[i])
        if low_i > upper_at * 1.003:
            breakout = "up"
            break
    
    if breakout is None:
        for i in range(anchor2[0], n):
            lower_at = _price_at(slope, lower_int, i)
            high_i = float(df["high"].iloc[i])
            if high_i < lower_at * 0.997:
                breakout = "down"
                break
    
    last_idx = n - 1
    last_close = float(df["close"].iloc[-1])
    upper_now = _price_at(slope, upper_int, last_idx)
    lower_now = _price_at(slope, lower_int, last_idx)
    position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0
    
    return {
        "direction": _direction(slope),
        "upper_line": {"slope": slope, "intercept": upper_int,
                       "points": upper_touches},
        "lower_line": {"slope": slope, "intercept": lower_int,
                       "points": lower_touches},
        "width_pct": width_pct,
        "price_position": position,
        "breakout": breakout,
        "second_channel": None,
        "predicted_channel": None,
        "touches_upper": len(upper_touches),
        "touches_lower": len(lower_touches),
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
    }
