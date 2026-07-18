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


def _adaptive_high(df):
    """Adaptive peak price: high, but body_top when:
    - wick > 3x body (spike noise), OR
    - wick > 10% of body_top price (long wick relative to price)
    For doji (body ≈ 0): average of high and close."""
    high = df["high"].values.astype(float)
    opn = df["open"].values.astype(float)
    close = df["close"].values.astype(float)
    body_top = np.maximum(opn, close)
    body_size = np.abs(close - opn)
    upper_wick = high - body_top
    # Doji detection: body < 0.1% of price
    is_doji = body_size < close * 0.001
    # Spike: upper wick > 3x body OR wick > 10% of body_top
    body_safe = np.where(body_size == 0, 1e10, body_size)
    is_spike_3x = upper_wick > 3.0 * body_safe
    is_spike_10pct = upper_wick > body_top * 0.10
    is_spike = is_spike_3x | is_spike_10pct
    result = high.copy()
    result = np.where(is_spike & ~is_doji, body_top, result)
    result = np.where(is_doji, (high + close) / 2.0, result)
    return result


def _adaptive_low(df):
    """Adaptive valley price: low, but body_bot when:
    - wick > 3x body (spike noise), OR
    - wick > 10% of body_bot price (long wick relative to price)
    For doji (body ≈ 0): average of low and close."""
    low = df["low"].values.astype(float)
    opn = df["open"].values.astype(float)
    close = df["close"].values.astype(float)
    body_bot = np.minimum(opn, close)
    body_size = np.abs(close - opn)
    lower_wick = body_bot - low
    is_doji = body_size < close * 0.001
    body_safe = np.where(body_size == 0, 1e10, body_size)
    is_spike_3x = lower_wick > 3.0 * body_safe
    is_spike_10pct = lower_wick > body_bot * 0.10
    is_spike = is_spike_3x | is_spike_10pct
    result = low.copy()
    result = np.where(is_spike & ~is_doji, body_bot, result)
    result = np.where(is_doji, (low + close) / 2.0, result)
    return result


# Keep old names as aliases for any external usage
_smart_high = _adaptive_high
_smart_low = _adaptive_low


def find_swing_highs(df, min_radius=1, max_radius=6, min_distance=3):
    """Hybrid peak finder: flexible radius (1-6) from AIAlisa + prominence sort.
    Uses adaptive prices (high, but body when wick > 3x body)."""
    prices = _adaptive_high(df)
    n = len(prices)
    candidates = []
    for i in range(1, n - 1):
        best_radius = 0
        for r in range(min_radius, min(max_radius + 1, min(i + 1, n - i))):
            left = prices[max(0, i - r):i]
            right = prices[i + 1:min(n, i + r + 1)]
            if len(left) == 0 or len(right) == 0:
                break
            if prices[i] < np.max(left) or prices[i] < np.max(right):
                break
            best_radius = r
        if best_radius < min_radius:
            continue
        # Require slope: at least one neighbor on each side strictly lower
        has_left_slope = any(prices[j] < prices[i] for j in range(max(0, i - best_radius), i))
        has_right_slope = any(prices[j] < prices[i] for j in range(i + 1, min(n, i + best_radius + 1)))
        if has_left_slope and has_right_slope:
            # Reject flat tops: if next candle has same price, skip
            if i + 1 < n and prices[i + 1] == prices[i]:
                continue
            candidates.append((i, float(prices[i]), best_radius))
    if not candidates:
        return []
    # Sort by price descending (prominence: highest first)
    candidates.sort(key=lambda x: -x[1])
    result = []
    for idx, price, _ in candidates:
        if all(abs(idx - r[0]) >= min_distance for r in result):
            result.append((idx, price))
    result.sort(key=lambda x: x[0])
    return result


def find_swing_highs_body(df, min_radius=1, max_radius=6, min_distance=3):
    """Same as find_swing_highs but uses body tops (max(open, close)) instead of adaptive highs."""
    prices = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    n = len(prices)
    candidates = []
    for i in range(1, n - 1):
        best_radius = 0
        for r in range(min_radius, min(max_radius + 1, min(i + 1, n - i))):
            left = prices[max(0, i - r):i]
            right = prices[i + 1:min(n, i + r + 1)]
            if len(left) == 0 or len(right) == 0:
                break
            if prices[i] < np.max(left) or prices[i] < np.max(right):
                break
            best_radius = r
        if best_radius < min_radius:
            continue
        has_left_slope = any(prices[j] < prices[i] for j in range(max(0, i - best_radius), i))
        has_right_slope = any(prices[j] < prices[i] for j in range(i + 1, min(n, i + best_radius + 1)))
        if has_left_slope and has_right_slope:
            if i + 1 < n and prices[i + 1] == prices[i]:
                continue
            candidates.append((i, float(prices[i]), best_radius))
    if not candidates:
        return []
    candidates.sort(key=lambda x: -x[1])
    result = []
    for idx, price, _ in candidates:
        if all(abs(idx - r[0]) >= min_distance for r in result):
            result.append((idx, price))
    result.sort(key=lambda x: x[0])
    return result


def find_swing_lows_body(df, min_radius=1, max_radius=6, min_distance=3):
    """Same as find_swing_lows but uses body bottoms (min(open, close)) instead of adaptive lows."""
    prices = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    n = len(prices)
    candidates = []
    for i in range(1, n - 1):
        best_radius = 0
        for r in range(min_radius, min(max_radius + 1, min(i + 1, n - i))):
            left = prices[max(0, i - r):i]
            right = prices[i + 1:min(n, i + r + 1)]
            if len(left) == 0 or len(right) == 0:
                break
            if prices[i] > np.min(left) or prices[i] > np.min(right):
                break
            best_radius = r
        if best_radius < min_radius:
            continue
        has_left_slope = any(prices[j] > prices[i] for j in range(max(0, i - best_radius), i))
        has_right_slope = any(prices[j] > prices[i] for j in range(i + 1, min(n, i + best_radius + 1)))
        if has_left_slope and has_right_slope:
            if i + 1 < n and prices[i + 1] == prices[i]:
                continue
            candidates.append((i, float(prices[i]), best_radius))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[1])
    result = []
    for idx, price, _ in candidates:
        if all(abs(idx - r[0]) >= min_distance for r in result):
            result.append((idx, price))
    result.sort(key=lambda x: x[0])
    return result


def find_swing_lows(df, min_radius=1, max_radius=6, min_distance=3):
    """Hybrid valley finder: flexible radius (1-6) + prominence sort.
    Uses adaptive prices (low, but body when wick > 3x body)."""
    prices = _adaptive_low(df)
    n = len(prices)
    candidates = []
    for i in range(1, n - 1):
        best_radius = 0
        for r in range(min_radius, min(max_radius + 1, min(i + 1, n - i))):
            left = prices[max(0, i - r):i]
            right = prices[i + 1:min(n, i + r + 1)]
            if len(left) == 0 or len(right) == 0:
                break
            if prices[i] > np.min(left) or prices[i] > np.min(right):
                break
            best_radius = r
        if best_radius < min_radius:
            continue
        has_left_slope = any(prices[j] > prices[i] for j in range(max(0, i - best_radius), i))
        has_right_slope = any(prices[j] > prices[i] for j in range(i + 1, min(n, i + best_radius + 1)))
        if has_left_slope and has_right_slope:
            if i + 1 < n and prices[i + 1] == prices[i]:
                continue
            candidates.append((i, float(prices[i]), best_radius))
    if not candidates:
        return []
    # Sort by price ascending (lowest first = most prominent valley)
    candidates.sort(key=lambda x: x[1])
    result = []
    for idx, price, _ in candidates:
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


def _interval_to_minutes(interval: str) -> int:
    """Convert interval string (1m, 5m, 1h, 4h, 1d, 1w, 1M) to minutes."""
    s = interval.strip().lower()
    if s.endswith("m") and s[:-1].isdigit():
        return int(s[:-1])
    elif s.endswith("h"):
        return int(s[:-1]) * 60
    elif s.endswith("d"):
        return int(s[:-1]) * 1440
    elif s.endswith("w"):
        return int(s[:-1]) * 10080
    elif s.upper().endswith("M"):
        return int(s[:-1]) * 43200  # ~30 days
    return 240  # default 4h


def _candles_in_days(interval: str, days: int) -> int:
    """How many candles fit in N days for a given interval."""
    mins = _interval_to_minutes(interval)
    return max(1, (days * 1440) // mins)


def detect_channel(df, interval="4h"):
    """
    Main channel detection — tries all algorithms, returns best + extras.
    
    Algorithm 1 (classic): highest peak → next peak right → descending channel
      Precondition: coin grew 20%+ from lowest body in last 30 days
    Algorithm 2 (recent highs): rightmost peaks → work backwards → steep/recent moves
    Algorithm 3 (body lows): 2 anchors on body bottoms, upper = same slope on highest body top
    
    Returns the best channel with 'extra_channels' list of other valid results.
    """
    MAX_WIDTH = 100.0
    all_results = []
    
    r1 = _detect_channel_v1(df, interval=interval)
    if r1 is not None and r1["width_pct"] <= MAX_WIDTH:
        if _validate_channel_bodies(df, r1):
            r1["algorithm"] = 1
            all_results.append(r1)
    
    r1_1 = _detect_channel_v1_1(df, interval=interval)
    if r1_1 is not None and r1_1["width_pct"] <= MAX_WIDTH:
        if _validate_channel_bodies(df, r1_1):
            r1_1["algorithm"] = 1.1
            all_results.append(r1_1)
    
    r2 = _detect_channel_v2(df)
    if r2 is not None:
        r2["algorithm"] = 2
        all_results.append(r2)
    
    r3 = _detect_channel_v3(df)
    if r3 is not None:
        r3["algorithm"] = 3
        all_results.append(r3)
    
    # Algo 4 family: priority 4 → 4.1
    r4 = _detect_channel_v4(df)
    if r4 is not None:
        r4["algorithm"] = 4
        all_results.append(r4)
    else:
        r4_1 = _detect_channel_v4_1(df)
        if r4_1 is not None:
            r4_1["algorithm"] = 4.1
            all_results.append(r4_1)
    
    # Algo 5 family: priority 5 → 5.1 → 5.2 (only first successful)
    r5 = _detect_channel_v5(df, interval=interval)
    if r5 is not None and _validate_channel_bodies(df, r5):
        r5["algorithm"] = 5
        all_results.append(r5)
    else:
        r5_1 = _detect_channel_v5_1(df, interval=interval)
        if r5_1 is not None and _validate_channel_bodies(df, r5_1):
            r5_1["algorithm"] = 5.1
            all_results.append(r5_1)
        else:
            r5_2 = _detect_channel_v5_2(df, interval=interval)
            if r5_2 is not None and _validate_channel_bodies(df, r5_2):
                r5_2["algorithm"] = 5.2
                all_results.append(r5_2)
    
    if not all_results:
        return None
    
    # Primary = first valid (priority order: 1, 2, 3)
    primary = all_results[0]
    
    # Filter extras: no duplicates vs primary or each other
    # Allow max 1 extra with different direction from primary
    n = len(df)
    extras = []
    has_opposite = False
    for ec in all_results[1:]:
        if _channels_similar(primary, ec, n):
            continue
        # Also check against already accepted extras
        dup_of_extra = False
        for accepted in extras:
            if _channels_similar(accepted, ec, n):
                dup_of_extra = True
                break
        if dup_of_extra:
            continue
        # Only one channel with opposite direction allowed
        if ec.get("direction") != primary.get("direction"):
            if has_opposite:
                continue
            has_opposite = True
        extras.append(ec)
    
    primary["extra_channels"] = extras
    return primary


def _channels_similar(ch1, ch2, n):
    """Check if two channels are too similar or one contains the other.
    Channels with different directions (ascending vs descending) are never duplicates.
    Checks overlap in the anchor region of both channels.
    If overlap >= 85% of EITHER channel → duplicate."""
    # Different directions → always allowed together
    if ch1.get("direction") != ch2.get("direction"):
        return False
    # Find the actual span where both channels are active
    a1_all = ch1.get("anchors", {})
    a2_all = ch2.get("anchors", {})
    a1_pts = a1_all.get("lower", []) + a1_all.get("upper", [])
    a2_pts = a2_all.get("lower", []) + a2_all.get("upper", [])
    
    if a1_pts and a2_pts:
        # Check from the later first anchor to the end
        start = max(min(int(a[0]) for a in a1_pts), min(int(a[0]) for a in a2_pts))
    else:
        start = n // 2
    
    # Check at 3 points in the active region
    span = n - start
    if span < 3:
        return False
    points = [start + span // 4, start + span // 2, start + 3 * span // 4]
    
    overlap_count = 0
    for p in points:
        if p >= n:
            continue
        u1 = _price_at(ch1["upper_line"]["slope"], ch1["upper_line"]["intercept"], p)
        l1 = _price_at(ch1["lower_line"]["slope"], ch1["lower_line"]["intercept"], p)
        u2 = _price_at(ch2["upper_line"]["slope"], ch2["upper_line"]["intercept"], p)
        l2 = _price_at(ch2["lower_line"]["slope"], ch2["lower_line"]["intercept"], p)
        
        h1 = u1 - l1
        h2 = u2 - l2
        if h1 <= 0 or h2 <= 0:
            continue
        
        overlap_top = min(u1, u2)
        overlap_bot = max(l1, l2)
        overlap = max(0, overlap_top - overlap_bot)
        
        frac1 = overlap / h1
        frac2 = overlap / h2
        
        # Duplicate if EITHER channel is 85%+ contained in the other
        if frac1 > 0.85 or frac2 > 0.85:
            overlap_count += 1
    
    return overlap_count >= 2


def _validate_channel_bodies(df, ch):
    """Validate that no candle body crosses channel lines from first anchor to current candle.
    Anchor candles themselves are excluded (they define the lines).
    If any other body goes above upper line or below lower line → channel is invalid."""
    n = len(df)
    body_tops = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    body_bots = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    
    all_anchors = ch["anchors"].get("lower", []) + ch["anchors"].get("upper", [])
    if not all_anchors:
        return True
    
    first_anchor = min(int(a[0]) for a in all_anchors)
    anchor_indices = set(int(a[0]) for a in all_anchors)
    
    us = ch["upper_line"]["slope"]
    ui = ch["upper_line"]["intercept"]
    ls = ch["lower_line"]["slope"]
    li = ch["lower_line"]["intercept"]
    
    # Check from first anchor to end, skip anchor candles themselves
    for i in range(first_anchor, n):
        if i in anchor_indices:
            continue
        upper_at = _price_at(us, ui, i)
        lower_at = _price_at(ls, li, i)
        if body_tops[i] > upper_at * 1.003:
            return False
        if body_bots[i] < lower_at * 0.997:
            return False
    
    return True


def _detect_channel_v1(df, interval="4h"):
    """
    Algorithm 1 — Alisa's original.
    From highest peak → next peak to the right → descending channel.
    Good for well-established channels across the full 200 candles.
    
    Precondition: coin must have grown 20%+ from the lowest candle body
    within the last 30 days to current price. If not — algo doesn't apply.
    """
    n = len(df)
    if n < 50:
        return None

    # =============================================
    # PRECONDITION: 20%+ growth from lowest body bottom in last 30 days
    # =============================================
    body_bottoms = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    candles_30d = _candles_in_days(interval, 30)
    window_start = max(0, n - candles_30d)
    recent_body_bots = body_bottoms[window_start:]
    min_body_price = float(np.min(recent_body_bots))
    current_price = float(df["close"].values[-1])
    if min_body_price <= 0:
        return None
    growth_pct = (current_price - min_body_price) / min_body_price * 100
    if growth_pct < 10.0:
        return None

    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)
    smart_lows = _smart_low(df)
    smart_highs = _smart_high(df)

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
    right_peaks = [(i, p) for i, p in relevant_highs 
                   if i > anchor1[0] and p < anchor1[1]]
    
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
    # Find the lowest BODY BOTTOM strictly between the two upper anchors.
    # If no dip found there, allow up to 10 candles RIGHT of anchor2 (toward current price).
    span_start = anchor1[0]
    
    # 1) Strict: between anchor1 and anchor2 inclusive
    region_strict = body_bottoms[span_start:anchor2[0] + 1]
    low_idx = None
    low_price = None
    if len(region_strict) > 0:
        min_rel = np.argmin(region_strict)
        low_idx = span_start + min_rel
        low_price = float(region_strict[min_rel])
    
    # 2) If no valid dip between anchors, extend 10 candles right of anchor2
    if low_idx is None or low_price >= min(anchor1[1], anchor2[1]):
        ext_start = anchor2[0] + 1
        ext_end = min(n, anchor2[0] + 11)
        region_ext = body_bottoms[ext_start:ext_end]
        if len(region_ext) > 0:
            ext_rel = np.argmin(region_ext)
            ext_price = float(region_ext[ext_rel])
            if low_idx is None or ext_price < low_price:
                low_idx = ext_start + ext_rel
                low_price = ext_price
    
    if low_idx is None:
        return None
    
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


def _adjust_lower_anchor(df, idx, price):
    """Check 7 candles left and right of a lower anchor.
    If a lower body bottom exists nearby, return that candle instead.
    Returns (new_idx, new_price) or original if no better found."""
    n = len(df)
    body_bots = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    best_idx = idx
    best_price = price
    for k in range(max(0, idx - 7), min(n, idx + 8)):
        if k == idx:
            continue
        if body_bots[k] < best_price:
            best_idx = k
            best_price = float(body_bots[k])
    return best_idx, best_price


def _anchor_price_high(df, idx):
    """Anchor price for highs: use high, but if wick > 10% of body_top → use body_top."""
    h = float(df["high"].iloc[idx])
    bt = max(float(df["open"].iloc[idx]), float(df["close"].iloc[idx]))
    if bt > 0 and (h - bt) > bt * 0.10:
        return bt
    return h


def _anchor_price_low(df, idx):
    """Anchor price for lows: use low, but if wick > 10% of body_bot → use body_bot."""
    l = float(df["low"].iloc[idx])
    bb = min(float(df["open"].iloc[idx]), float(df["close"].iloc[idx]))
    if bb > 0 and (bb - l) > bb * 0.10:
        return bb
    return l


def _detect_channel_v1_1(df, interval="4h"):
    """
    Algorithm 1.1 — Same as Algorithm 1, but upper line built on BODY TOPS
    (max(open, close)) instead of swing highs (adaptive high/wicks).
    Everything else identical: precondition 10%+ growth, lower line on body bottoms, etc.
    """
    n = len(df)
    if n < 50:
        return None

    # PRECONDITION: 10%+ growth from lowest body bottom in last 30 days
    body_bottoms = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    candles_30d = _candles_in_days(interval, 30)
    window_start = max(0, n - candles_30d)
    recent_body_bots = body_bottoms[window_start:]
    min_body_price = float(np.min(recent_body_bots))
    current_price = float(df["close"].values[-1])
    if min_body_price <= 0:
        return None
    growth_pct = (current_price - min_body_price) / min_body_price * 100
    if growth_pct < 10.0:
        return None

    # Use body-top swing highs for upper line
    swing_highs_b = find_swing_highs_body(df)
    swing_lows = find_swing_lows(df)
    smart_lows = _smart_low(df)
    smart_highs = _smart_high(df)

    if len(swing_highs_b) < 2:
        return None

    # STEP 1: Highest body-top peak
    lookback_start = max(0, n - 200)
    relevant_highs = [(i, p) for i, p in swing_highs_b if i >= lookback_start]
    if len(relevant_highs) < 2:
        return None
    anchor1 = max(relevant_highs, key=lambda x: x[1])

    # STEP 2: Next body-top peak to the RIGHT, lower
    right_peaks = [(i, p) for i, p in relevant_highs
                   if i > anchor1[0] and p < anchor1[1]]
    if not right_peaks:
        return None
    anchor2 = max(right_peaks, key=lambda x: x[1])

    # STEP 3: Upper line through body-top anchors
    slope, upper_int = _log_line(anchor1[0], anchor1[1], anchor2[0], anchor2[1])
    if abs(slope) > 0.03:
        return None
    upper_touches = _touches(swing_highs_b, slope, upper_int, 0.015)

    # STEP 3b: If 4+ touches AND channel too wide, rebuild from 2 closest
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
        sorted_by_idx = sorted(upper_touches, key=lambda x: x[0], reverse=True)
        new_a2 = sorted_by_idx[0]
        new_a1 = sorted_by_idx[1]
        if new_a1[0] > new_a2[0]:
            new_a1, new_a2 = new_a2, new_a1
        anchor1 = new_a1
        anchor2 = new_a2
        slope, upper_int = _log_line(anchor1[0], anchor1[1], anchor2[0], anchor2[1])
        if abs(slope) > 0.03:
            return None
        upper_touches = _touches(swing_highs_b, slope, upper_int, 0.015)

    # STEP 4: Lower line — same slope, lowest body bottom
    # Strict: between anchors. If no dip → +10 candles right of anchor2.
    span_start = anchor1[0]
    
    region_strict = body_bottoms[span_start:anchor2[0] + 1]
    low_idx = None
    low_price = None
    if len(region_strict) > 0:
        min_rel = np.argmin(region_strict)
        low_idx = span_start + min_rel
        low_price = float(region_strict[min_rel])
    
    if low_idx is None or low_price >= min(anchor1[1], anchor2[1]):
        ext_start = anchor2[0] + 1
        ext_end = min(n, anchor2[0] + 11)
        region_ext = body_bottoms[ext_start:ext_end]
        if len(region_ext) > 0:
            ext_rel = np.argmin(region_ext)
            ext_price = float(region_ext[ext_rel])
            if low_idx is None or ext_price < low_price:
                low_idx = ext_start + ext_rel
                low_price = ext_price
    
    if low_idx is None:
        return None
    
    lower_int = np.log(low_price) - slope * low_idx

    # Validate width
    mid = (anchor1[0] + anchor2[0]) // 2
    um = _price_at(slope, upper_int, mid)
    lm = _price_at(slope, lower_int, mid)
    if lm >= um:
        return None
    width_pct = (um - lm) / lm * 100
    if width_pct < 1.0:
        return None
    lower_touches = _touches(swing_lows, slope, lower_int, 0.015)

    # STEP 5: Break point
    break_idx = None
    breakout = None
    search_start = low_idx
    for i in range(search_start, n):
        lower_at = _price_at(slope, lower_int, i)
        high_i = float(df["high"].iloc[i])
        if high_i < lower_at * 0.997:
            break_idx = i
            breakout = "down"
            break
    if break_idx is None:
        for i in range(anchor2[0], n):
            upper_at = _price_at(slope, upper_int, i)
            low_i = float(df["low"].iloc[i])
            if low_i > upper_at * 1.003:
                break_idx = i
                breakout = "up"
                break

    # STEP 6: CH2 after break
    second_channel = None
    if break_idx is not None and breakout == "down":
        ch2_a1 = anchor2
        post_peaks = [(i, p) for i, p in swing_highs_b
                      if i > ch2_a1[0] and (i - ch2_a1[0]) >= 5]
        if post_peaks:
            ch2_a2 = max(post_peaks, key=lambda x: x[0])
            ch2_slope, ch2_upper_int = _log_line(ch2_a1[0], ch2_a1[1], ch2_a2[0], ch2_a2[1])
            if abs(ch2_slope) <= 0.03:
                ch2_upper_touches = _touches(swing_highs_b, ch2_slope, ch2_upper_int, 0.015)
                ch1_lower_touches = _touches(swing_lows, slope, lower_int, 0.02)
                ch2_low_anchor = None
                for ti, tp in reversed(ch1_lower_touches):
                    if ti >= break_idx:
                        continue
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
                    region = smart_lows[max(0, ch2_a1[0]):min(n, break_idx + 5)]
                    if len(region) > 0:
                        ri = np.argmin(region)
                        ch2_low_anchor = (ch2_a1[0] + ri, float(region[ri]))
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

    # Result
    last_idx = n - 1
    last_close = float(df["close"].iloc[-1])
    upper_now = _price_at(slope, upper_int, last_idx)
    lower_now = _price_at(slope, lower_int, last_idx)
    position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0

    # Use regular swing_highs for the result (for chart drawing compatibility)
    swing_highs_regular = find_swing_highs(df)

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
        "swing_highs": swing_highs_regular,
        "swing_lows": swing_lows,
        "anchors": {
            "upper": [anchor1, anchor2],
            "lower": [(low_idx, low_price)],
        },
    }


def _detect_channel_v2(df):
    """
    Algorithm 2 — Recent peaks fallback.
    Start from the rightmost (most recent) swing highs, try pairs working backwards.
    Good for steep/recent channels that Algorithm 1 misses (e.g. sharp dumps).
    
    Rules:
    - Anchor prices: high, but body_top if wick > 10% of body_top
    - Upper line must NOT intersect candle bodies between anchors
    - Lower line = same slope, on lowest body bottom between anchors
    - NO candle body may protrude below lower line (between anchors AND after)
    - If bodies protrude → skip this pair, try next
    - Min 10 candles between anchors, max width 100%
    """
    n = len(df)
    if n < 50:
        return None
    
    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)
    
    if len(swing_highs) < 2:
        return None
    
    body_tops = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    body_bots = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    
    # Sort peaks by index descending (most recent first)
    # Exclude current and previous candle as anchors
    sorted_peaks = sorted(
        [(i, p) for i, p in swing_highs if i < n - 2],
        key=lambda x: x[0], reverse=True
    )
    
    for i in range(len(sorted_peaks)):
        for j in range(i + 1, len(sorted_peaks)):
            a2_raw = sorted_peaks[i]  # right (more recent)
            a1_raw = sorted_peaks[j]  # left (older)
            
            # Ensure a1 is left of a2
            if a1_raw[0] > a2_raw[0]:
                a1_raw, a2_raw = a2_raw, a1_raw
            
            # Min distance between anchors
            if a2_raw[0] - a1_raw[0] < 10:
                continue
            
            # Apply 10% wick rule for anchor prices
            a1_price = _anchor_price_high(df, a1_raw[0])
            a2_price = _anchor_price_high(df, a2_raw[0])
            a1 = (a1_raw[0], a1_price)
            a2 = (a2_raw[0], a2_price)
            
            # Must be descending (a1 higher than a2)
            if a1[1] <= a2[1]:
                continue
            
            slope, upper_int = _log_line(a1[0], a1[1], a2[0], a2[1])
            
            # Slope sanity
            if abs(slope) > 0.03:
                continue
            
            span_s = a1[0]
            span_e = a2[0] + 1
            
            # Upper line must NOT intersect candle bodies between anchors
            upper_ok = True
            for k in range(span_s, min(n, span_e)):
                u_at = _price_at(slope, upper_int, k)
                if u_at < body_tops[k] and u_at > body_bots[k]:
                    upper_ok = False
                    break
            if not upper_ok:
                continue
            
            # Lower line: same slope, on lowest body bottom between anchors
            region = body_bots[span_s:min(n, span_e)]
            if len(region) == 0:
                continue
            li = np.argmin(region)
            low_idx = span_s + li
            low_price = float(region[li])
            
            # Check 7 candles around for a lower body bottom (try adjusted, fallback original)
            orig_low_idx, orig_low_price = low_idx, low_price
            adj_idx, adj_price = _adjust_lower_anchor(df, low_idx, low_price)
            low_candidates = []
            if adj_idx != low_idx:
                low_candidates.append((adj_idx, adj_price))
            low_candidates.append((orig_low_idx, orig_low_price))
            
            best_lower = None
            for lc_idx, lc_price in low_candidates:
                lc_int = np.log(lc_price) - slope * lc_idx
                lc_ok = True
                for k in range(span_s, min(n, span_e)):
                    l_at = _price_at(slope, lc_int, k)
                    if l_at > body_bots[k] and l_at < body_tops[k]:
                        lc_ok = False
                        break
                if lc_ok:
                    best_lower = (lc_idx, lc_price, lc_int)
                    break
            
            if best_lower is None:
                continue
            low_idx, low_price, lower_int = best_lower
            
            # Width check
            mid = (a1[0] + a2[0]) // 2
            um = _price_at(slope, upper_int, mid)
            lm = _price_at(slope, lower_int, mid)
            if lm >= um:
                continue
            width_pct = (um - lm) / lm * 100
            if width_pct < 1.0 or width_pct > 100.0:
                continue
            
            # Validate bodies from 7 candles before anchors to END of data
            # If any body protrudes beyond channel lines → skip this pair
            full_valid = True
            check_start = max(0, span_s - 7)
            for k in range(check_start, n):
                u_at = _price_at(slope, upper_int, k)
                l_at = _price_at(slope, lower_int, k)
                # Body above upper line
                if body_tops[k] > u_at * 1.003 and body_bots[k] > u_at:
                    full_valid = False
                    break
                # Body below lower line
                if body_bots[k] < l_at * 0.997 and body_tops[k] < l_at:
                    full_valid = False
                    break
            if not full_valid:
                continue
            
            upper_touches = _touches(swing_highs, slope, upper_int, 0.015)
            lower_touches = _touches(swing_lows, slope, lower_int, 0.015)
            
            # Break detection
            break_idx = None
            breakout = None
            for bi in range(low_idx, n):
                lower_at = _price_at(slope, lower_int, bi)
                high_i = float(df["high"].iloc[bi])
                if high_i < lower_at * 0.997:
                    break_idx = bi
                    breakout = "down"
                    break
            if break_idx is None:
                for bi in range(a2[0], n):
                    upper_at = _price_at(slope, upper_int, bi)
                    low_i = float(df["low"].iloc[bi])
                    if low_i > upper_at * 1.003:
                        break_idx = bi
                        breakout = "up"
                        break
            
            # CH2 after break
            second_channel = None
            if break_idx is not None and breakout == "down":
                ch2_a1 = a2  # shared point
                post_peaks = [(pi, pp) for pi, pp in swing_highs
                              if pi > ch2_a1[0] and (pi - ch2_a1[0]) >= 5]
                if post_peaks:
                    ch2_a2 = max(post_peaks, key=lambda x: x[0])
                    ch2_slope, ch2_ui = _log_line(ch2_a1[0], ch2_a1[1], ch2_a2[0], ch2_a2[1])
                    if abs(ch2_slope) <= 0.03:
                        ch2_ss = ch2_a1[0]
                        ch2_se = min(n, ch2_a2[0] + 1)
                        ch2_region = body_bots[ch2_ss:ch2_se]
                        if len(ch2_region) > 0:
                            ch2_li = np.argmin(ch2_region)
                            ch2_low_price = float(ch2_region[ch2_li])
                            ch2_low_idx = ch2_ss + ch2_li
                            ch2_li_int = np.log(ch2_low_price) - ch2_slope * ch2_low_idx
                            # Shift lower if it cuts bodies
                            for k in range(ch2_ss, ch2_se):
                                l_at = _price_at(ch2_slope, ch2_li_int, k)
                                if l_at > body_bots[k] and l_at < body_tops[k]:
                                    ch2_li_int = np.log(body_bots[k]) - ch2_slope * k
                            mid2 = (ch2_a1[0] + ch2_a2[0]) // 2
                            um2 = _price_at(ch2_slope, ch2_ui, mid2)
                            lm2 = _price_at(ch2_slope, ch2_li_int, mid2)
                            if um2 > lm2:
                                ch2_w = (um2 - lm2) / lm2 * 100
                                ch2_ut = _touches(swing_highs, ch2_slope, ch2_ui, 0.015)
                                ch2_lt = _touches(swing_lows, ch2_slope, ch2_li_int, 0.015)
                                li_n = n - 1
                                lc = float(df["close"].iloc[-1])
                                u2n = _price_at(ch2_slope, ch2_ui, li_n)
                                l2n = _price_at(ch2_slope, ch2_li_int, li_n)
                                p2 = (lc - l2n) / (u2n - l2n) * 100 if u2n > l2n else 50
                                second_channel = {
                                    "direction": _direction(ch2_slope),
                                    "upper_line": {"slope": ch2_slope, "intercept": ch2_ui, "points": ch2_ut},
                                    "lower_line": {"slope": ch2_slope, "intercept": ch2_li_int, "points": ch2_lt},
                                    "width_pct": ch2_w, "price_position": p2,
                                    "touches_upper": len(ch2_ut), "touches_lower": len(ch2_lt),
                                    "anchors": {"upper": [ch2_a1, ch2_a2], "lower": [(ch2_low_idx, ch2_low_price)]},
                                }
            
            last_idx = n - 1
            last_close = float(df["close"].iloc[-1])
            upper_now = _price_at(slope, upper_int, last_idx)
            lower_now = _price_at(slope, lower_int, last_idx)
            position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0
            
            return {
                "direction": _direction(slope),
                "upper_line": {"slope": slope, "intercept": upper_int, "points": upper_touches},
                "lower_line": {"slope": slope, "intercept": lower_int, "points": lower_touches},
                "width_pct": width_pct, "price_position": position,
                "breakout": breakout, "second_channel": second_channel,
                "predicted_channel": None,
                "touches_upper": len(upper_touches), "touches_lower": len(lower_touches),
                "swing_highs": swing_highs, "swing_lows": swing_lows,
                "anchors": {"upper": [a1, a2], "lower": [(low_idx, low_price)]},
            }
    
    return None


def _detect_channel_v3(df):
    """
    Algorithm 3 — Body-based channel from lows.
    When algo 1 & 2 fail (upper lines pierce bodies), build channel from below:
    
    1. Find 2 anchor points on body BOTTOMS (swing lows, using body prices)
    2. Lower line through these 2 anchors
    3. Upper line = same slope, on highest body TOP between anchors
    4. Validate: no body protrudes above upper line or below lower line
    5. Start from rightmost (most recent) pairs, work backwards
    """
    n = len(df)
    if n < 50:
        return None
    
    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)
    
    if len(swing_lows) < 2:
        return None
    
    body_tops = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    body_bots = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    
    # Build swing lows with body prices (apply 10% wick rule)
    sorted_lows = sorted(swing_lows, key=lambda x: x[0], reverse=True)
    
    for i in range(len(sorted_lows)):
        for j in range(i + 1, len(sorted_lows)):
            a2_raw = sorted_lows[i]   # right (recent)
            a1_raw = sorted_lows[j]   # left (older)
            
            if a1_raw[0] > a2_raw[0]:
                a1_raw, a2_raw = a2_raw, a1_raw
            
            if a2_raw[0] - a1_raw[0] < 10:
                continue
            
            # Anchor prices = ALWAYS body bottoms in algo 3
            a1_price = float(min(df["open"].iloc[a1_raw[0]], df["close"].iloc[a1_raw[0]]))
            a2_price = float(min(df["open"].iloc[a2_raw[0]], df["close"].iloc[a2_raw[0]]))
            
            # Check 7 candles around each anchor for lower body bottom (try adjusted, fallback original)
            a1_adj_idx, a1_adj_price = _adjust_lower_anchor(df, a1_raw[0], a1_price)
            a2_adj_idx, a2_adj_price = _adjust_lower_anchor(df, a2_raw[0], a2_price)
            
            # Try adjusted first, fallback to original
            anchor_candidates = []
            if a1_adj_idx != a1_raw[0] or a2_adj_idx != a2_raw[0]:
                anchor_candidates.append(((a1_adj_idx, a1_adj_price), (a2_adj_idx, a2_adj_price)))
            anchor_candidates.append(((a1_raw[0], a1_price), (a2_raw[0], a2_price)))
            
            found_pair = False
            for a1_cand, a2_cand in anchor_candidates:
                test_slope, test_li = _log_line(a1_cand[0], a1_cand[1], a2_cand[0], a2_cand[1])
                if test_slope < 0 and abs(test_slope) <= 0.03:
                    a1 = a1_cand
                    a2 = a2_cand
                    slope = test_slope
                    lower_int = test_li
                    found_pair = True
                    break
            
            if not found_pair:
                continue
            
            # Algo 3 = descending only
            if slope >= 0:
                continue
            if abs(slope) > 0.03:
                continue
            
            span_s = a1[0]
            span_e = min(n, a2[0] + 1)
            
            # Lower line must NOT intersect candle bodies between anchors
            lower_ok = True
            for k in range(span_s, span_e):
                l_at = _price_at(slope, lower_int, k)
                if l_at > body_bots[k] and l_at < body_tops[k]:
                    lower_ok = False
                    break
            if not lower_ok:
                continue
            
            # Upper line: same slope, on highest body TOP between anchors
            region_tops = body_tops[span_s:span_e]
            if len(region_tops) == 0:
                continue
            hi = np.argmax(region_tops)
            high_idx = span_s + hi
            high_price = float(region_tops[hi])
            upper_int = np.log(high_price) - slope * high_idx
            
            # Upper line must NOT intersect candle bodies between anchors
            upper_ok = True
            for k in range(span_s, span_e):
                u_at = _price_at(slope, upper_int, k)
                if u_at < body_tops[k] and u_at > body_bots[k]:
                    upper_ok = False
                    break
            if not upper_ok:
                continue
            
            # Width check
            mid = (a1[0] + a2[0]) // 2
            um = _price_at(slope, upper_int, mid)
            lm = _price_at(slope, lower_int, mid)
            if lm >= um:
                continue
            width_pct = (um - lm) / lm * 100
            if width_pct < 1.0 or width_pct > 100.0:
                continue
            
            # Validate bodies from 7 candles before anchors to END of data
            # If any body fully outside channel → skip this pair
            full_valid = True
            check_start = max(0, span_s - 7)
            for k in range(check_start, n):
                u_at = _price_at(slope, upper_int, k)
                l_at = _price_at(slope, lower_int, k)
                if body_tops[k] > u_at * 1.003 and body_bots[k] > u_at:
                    full_valid = False
                    break
                if body_bots[k] < l_at * 0.997 and body_tops[k] < l_at:
                    full_valid = False
                    break
            if not full_valid:
                continue
            
            upper_touches = _touches(swing_highs, slope, upper_int, 0.015)
            lower_touches = _touches(swing_lows, slope, lower_int, 0.015)
            
            # Break detection
            break_idx = None
            breakout = None
            # Break down: body below lower line
            for bi in range(a2[0] + 1, n):
                if body_bots[bi] < _price_at(slope, lower_int, bi) * 0.997:
                    break_idx = bi
                    breakout = "down"
                    break
            # Break up: body above upper line
            if break_idx is None:
                for bi in range(a2[0] + 1, n):
                    if body_tops[bi] > _price_at(slope, upper_int, bi) * 1.003:
                        break_idx = bi
                        breakout = "up"
                        break
            
            # Filter touches to before break
            if break_idx is not None:
                upper_touches = [(ti, tp) for ti, tp in upper_touches if ti <= break_idx]
                lower_touches = [(ti, tp) for ti, tp in lower_touches if ti <= break_idx]
            
            last_idx = n - 1
            last_close = float(df["close"].iloc[-1])
            upper_now = _price_at(slope, upper_int, last_idx)
            lower_now = _price_at(slope, lower_int, last_idx)
            position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0
            
            return {
                "direction": _direction(slope),
                "upper_line": {"slope": slope, "intercept": upper_int, "points": upper_touches},
                "lower_line": {"slope": slope, "intercept": lower_int, "points": lower_touches},
                "width_pct": width_pct, "price_position": position,
                "breakout": breakout, "second_channel": None,
                "predicted_channel": None,
                "touches_upper": len(upper_touches), "touches_lower": len(lower_touches),
                "swing_highs": swing_highs, "swing_lows": swing_lows,
                "anchors": {"lower": [a1, a2], "upper": [(high_idx, high_price)]},
            }
    
    return None


def _detect_channel_v4(df):
    """
    Algorithm 4 — Ascending channel from body lows (relaxed).
    Same as algo 4.1 but with relaxed validation: channel is valid if
    85%+ of candles from first anchor to end have bodies inside the channel.
    This finds broader channels that cover most of the price action.
    """
    n = len(df)
    if n < 50:
        return None
    
    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)
    
    if len(swing_lows) < 2:
        return None
    
    body_tops = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    body_bots = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    highs = df["high"].values.astype(float)
    
    # Sort lows by index ascending (farthest first) — covers more candles for 85% rule
    # Exclude current and previous candle
    sorted_lows = sorted(
        [(i, p) for i, p in swing_lows if i < n - 2],
        key=lambda x: x[0]
    )
    
    for i in range(len(sorted_lows) - 1):
        for j in range(i + 1, len(sorted_lows)):
            a_raw = sorted_lows[i]   # left (older) = point A (lower)
            b_raw = sorted_lows[j]   # right (recent) = point B (higher)
            
            # Min 15 candles between anchors
            if b_raw[0] - a_raw[0] < 15:
                continue
            
            # Anchor prices = ALWAYS body bottoms
            a_orig_price = float(body_bots[a_raw[0]])
            b_orig_price = float(body_bots[b_raw[0]])
            
            # Check 7 candles around each anchor for lower body bottom
            a_adj_idx, a_adj_price = _adjust_lower_anchor(df, a_raw[0], a_orig_price)
            b_adj_idx, b_adj_price = _adjust_lower_anchor(df, b_raw[0], b_orig_price)
            
            # Try adjusted first, fallback to original
            candidates = []
            if a_adj_idx != a_raw[0] or b_adj_idx != b_raw[0]:
                candidates.append((a_adj_idx, a_adj_price, b_adj_idx, b_adj_price))
            candidates.append((a_raw[0], a_orig_price, b_raw[0], b_orig_price))
            
            found = False
            for a_idx, a_price, b_idx, b_price in candidates:
                if b_idx - a_idx < 15:
                    continue
                if a_price >= b_price:
                    continue
                a = (a_idx, a_price)
                b = (b_idx, b_price)
                found = True
                break
            
            if not found:
                continue
            
            slope, lower_int = _log_line(a[0], a[1], b[0], b[1])
            
            # Must be ascending
            if slope <= 0:
                continue
            if abs(slope) > 0.03:
                continue
            
            span_s = a[0]
            span_e = b[0] + 1
            
            # Lower line: between anchors, line must NOT intersect candle bodies
            # Exception: ±1 candle from each anchor is allowed
            anchor_tolerance = {a[0] - 1, a[0], a[0] + 1, b[0] - 1, b[0], b[0] + 1}
            lower_ok = True
            for k in range(span_s, min(n, span_e)):
                if k in anchor_tolerance:
                    continue
                l_at = _price_at(slope, lower_int, k)
                if l_at > body_bots[k] and l_at < body_tops[k]:
                    lower_ok = False
                    break
            if not lower_ok:
                continue
            
            # Upper line: same slope, on highest body TOP between anchors
            # Apply wick rule: if wick > 2x body OR > 10% → use body_top
            region_tops = body_tops[span_s:min(n, span_e)]
            if len(region_tops) == 0:
                continue
            hi = np.argmax(region_tops)
            high_idx = span_s + hi
            high_price = float(region_tops[hi])
            
            # Check if we should use high instead of body_top
            actual_high = float(highs[high_idx])
            wick = actual_high - high_price
            body_size = float(body_tops[high_idx] - body_bots[high_idx])
            body_safe = body_size if body_size > 0 else 1e10
            
            # Only use high if wick is small (< 2x body AND < 10% of body_top)
            if wick <= body_safe * 2.0 and wick <= high_price * 0.10:
                high_price = actual_high
            # else: keep body_top (wick too long)
            
            upper_int = np.log(high_price) - slope * high_idx
            
            # Upper line must NOT intersect candle bodies between anchors
            upper_ok = True
            for k in range(span_s, min(n, span_e)):
                u_at = _price_at(slope, upper_int, k)
                if u_at < body_tops[k] and u_at > body_bots[k]:
                    upper_ok = False
                    break
            if not upper_ok:
                continue
            
            # Width check
            mid = (a[0] + b[0]) // 2
            um = _price_at(slope, upper_int, mid)
            lm = _price_at(slope, lower_int, mid)
            if lm >= um:
                continue
            width_pct = (um - lm) / lm * 100
            if width_pct < 1.0 or width_pct > 50.0:
                continue
            
            # Validation:
            # 1) From first anchor to end: lines CANNOT cross candle bodies (hard rule)
            # 2) 85%+ of ALL candles (from start of chart) must be inside the channel
            #    The 15% allowance covers candles at the beginning before the channel
            
            # Step 1: no crossing from first anchor onward
            lines_cross = False
            for k in range(span_s, n):
                u_at = _price_at(slope, upper_int, k)
                l_at = _price_at(slope, lower_int, k)
                bt = body_tops[k]
                bb = body_bots[k]
                if u_at > bb and u_at < bt:
                    lines_cross = True
                    break
                if l_at > bb and l_at < bt:
                    lines_cross = True
                    break
            if lines_cross:
                continue
            
            # Step 2: 85%+ of all candles must be fully inside
            outside_count = 0
            for k in range(n):
                u_at = _price_at(slope, upper_int, k)
                l_at = _price_at(slope, lower_int, k)
                if body_tops[k] > u_at * 1.003 or body_bots[k] < l_at * 0.997:
                    outside_count += 1
            if (n - outside_count) / n < 0.85:
                continue
            
            upper_touches = _touches(swing_highs, slope, upper_int, 0.015)
            lower_touches = _touches(swing_lows, slope, lower_int, 0.015)
            
            # Break detection
            break_idx = None
            breakout = None
            for bi in range(b[0] + 1, n):
                if body_tops[bi] > _price_at(slope, upper_int, bi) * 1.003:
                    break_idx = bi
                    breakout = "up"
                    break
            if break_idx is None:
                for bi in range(b[0] + 1, n):
                    if body_bots[bi] < _price_at(slope, lower_int, bi) * 0.997:
                        break_idx = bi
                        breakout = "down"
                        break
            
            if break_idx is not None:
                upper_touches = [(ti, tp) for ti, tp in upper_touches if ti <= break_idx]
                lower_touches = [(ti, tp) for ti, tp in lower_touches if ti <= break_idx]
            
            last_idx = n - 1
            last_close = float(df["close"].iloc[-1])
            upper_now = _price_at(slope, upper_int, last_idx)
            lower_now = _price_at(slope, lower_int, last_idx)
            position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0
            
            return {
                "direction": _direction(slope),
                "upper_line": {"slope": slope, "intercept": upper_int, "points": upper_touches},
                "lower_line": {"slope": slope, "intercept": lower_int, "points": lower_touches},
                "width_pct": width_pct, "price_position": position,
                "breakout": breakout, "second_channel": None,
                "predicted_channel": None,
                "touches_upper": len(upper_touches), "touches_lower": len(lower_touches),
                "swing_highs": swing_highs, "swing_lows": swing_lows,
                "anchors": {"lower": [a, b], "upper": [(high_idx, high_price)]},
            }
    
    return None


def _detect_channel_v4_1(df):
    """
    Algorithm 4.1 — Ascending channel from body lows (strict).
    Same as algo 4 but with STRICT validation: zero body violations allowed.
    Fallback when algo 4 (relaxed 85%) finds a channel but strict is preferred.
    """
    n = len(df)
    if n < 50:
        return None
    
    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)
    
    if len(swing_lows) < 2:
        return None
    
    body_tops = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    body_bots = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    highs = df["high"].values.astype(float)
    
    sorted_lows = sorted(
        [(i, p) for i, p in swing_lows if i < n - 2],
        key=lambda x: x[0], reverse=True
    )
    
    for i in range(len(sorted_lows)):
        for j in range(i + 1, len(sorted_lows)):
            b_raw = sorted_lows[i]
            a_raw = sorted_lows[j]
            
            if a_raw[0] > b_raw[0]:
                a_raw, b_raw = b_raw, a_raw
            
            if b_raw[0] - a_raw[0] < 15:
                continue
            
            a_orig_price = float(body_bots[a_raw[0]])
            b_orig_price = float(body_bots[b_raw[0]])
            
            a_adj_idx, a_adj_price = _adjust_lower_anchor(df, a_raw[0], a_orig_price)
            b_adj_idx, b_adj_price = _adjust_lower_anchor(df, b_raw[0], b_orig_price)
            
            candidates = []
            if a_adj_idx != a_raw[0] or b_adj_idx != b_raw[0]:
                candidates.append((a_adj_idx, a_adj_price, b_adj_idx, b_adj_price))
            candidates.append((a_raw[0], a_orig_price, b_raw[0], b_orig_price))
            
            found = False
            for a_idx, a_price, b_idx, b_price in candidates:
                if b_idx - a_idx < 15:
                    continue
                if a_price >= b_price:
                    continue
                a = (a_idx, a_price)
                b = (b_idx, b_price)
                found = True
                break
            
            if not found:
                continue
            
            slope, lower_int = _log_line(a[0], a[1], b[0], b[1])
            
            if slope <= 0 or abs(slope) > 0.03:
                continue
            
            span_s = a[0]
            span_e = b[0] + 1
            
            anchor_tolerance = {a[0] - 1, a[0], a[0] + 1, b[0] - 1, b[0], b[0] + 1}
            lower_ok = True
            for k in range(span_s, min(n, span_e)):
                if k in anchor_tolerance:
                    continue
                l_at = _price_at(slope, lower_int, k)
                if l_at > body_bots[k] and l_at < body_tops[k]:
                    lower_ok = False
                    break
            if not lower_ok:
                continue
            
            region_tops = body_tops[span_s:min(n, span_e)]
            if len(region_tops) == 0:
                continue
            hi = np.argmax(region_tops)
            high_idx = span_s + hi
            high_price = float(region_tops[hi])
            
            actual_high = float(highs[high_idx])
            wick = actual_high - high_price
            body_size = float(body_tops[high_idx] - body_bots[high_idx])
            body_safe = body_size if body_size > 0 else 1e10
            
            if wick <= body_safe * 2.0 and wick <= high_price * 0.10:
                high_price = actual_high
            
            upper_int = np.log(high_price) - slope * high_idx
            
            upper_ok = True
            for k in range(span_s, min(n, span_e)):
                u_at = _price_at(slope, upper_int, k)
                if u_at < body_tops[k] and u_at > body_bots[k]:
                    upper_ok = False
                    break
            if not upper_ok:
                continue
            
            mid = (a[0] + b[0]) // 2
            um = _price_at(slope, upper_int, mid)
            lm = _price_at(slope, lower_int, mid)
            if lm >= um:
                continue
            width_pct = (um - lm) / lm * 100
            if width_pct < 1.0 or width_pct > 100.0:
                continue
            
            # STRICT validation: zero body violations
            full_valid = True
            check_start = max(0, span_s - 7)
            for k in range(check_start, n):
                u_at = _price_at(slope, upper_int, k)
                l_at = _price_at(slope, lower_int, k)
                if body_tops[k] > u_at * 1.003 and body_bots[k] > u_at:
                    full_valid = False
                    break
                if body_bots[k] < l_at * 0.997 and body_tops[k] < l_at:
                    full_valid = False
                    break
            if not full_valid:
                continue
            
            upper_touches = _touches(swing_highs, slope, upper_int, 0.015)
            lower_touches = _touches(swing_lows, slope, lower_int, 0.015)
            
            break_idx = None
            breakout = None
            for bi in range(b[0] + 1, n):
                if body_tops[bi] > _price_at(slope, upper_int, bi) * 1.003:
                    break_idx = bi
                    breakout = "up"
                    break
            if break_idx is None:
                for bi in range(b[0] + 1, n):
                    if body_bots[bi] < _price_at(slope, lower_int, bi) * 0.997:
                        break_idx = bi
                        breakout = "down"
                        break
            
            if break_idx is not None:
                upper_touches = [(ti, tp) for ti, tp in upper_touches if ti <= break_idx]
                lower_touches = [(ti, tp) for ti, tp in lower_touches if ti <= break_idx]
            
            last_idx = n - 1
            last_close = float(df["close"].iloc[-1])
            upper_now = _price_at(slope, upper_int, last_idx)
            lower_now = _price_at(slope, lower_int, last_idx)
            position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0
            
            return {
                "direction": _direction(slope),
                "upper_line": {"slope": slope, "intercept": upper_int, "points": upper_touches},
                "lower_line": {"slope": slope, "intercept": lower_int, "points": lower_touches},
                "width_pct": width_pct, "price_position": position,
                "breakout": breakout, "second_channel": None,
                "predicted_channel": None,
                "touches_upper": len(upper_touches), "touches_lower": len(lower_touches),
                "swing_highs": swing_highs, "swing_lows": swing_lows,
                "anchors": {"lower": [a, b], "upper": [(high_idx, high_price)]},
            }
    
    return None


def _build_channel_from_lows(df, anchor1, anchor2, swing_lows_list, use_bodies=True):
    """
    Helper: build ascending channel from two lower anchors.
    anchor1 (LEFT) lower price, anchor2 (RIGHT, toward price) higher price.
    Upper anchor = highest body top (use_bodies=True) or high (False) between anchors.
    Returns channel dict or None.
    """
    n = len(df)
    slope, lower_int = _log_line(anchor1[0], anchor1[1], anchor2[0], anchor2[1])

    # Must be ascending
    if slope <= 0 or abs(slope) > 0.03:
        return None

    body_tops = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    prices_for_upper = body_tops if use_bodies else df["high"].values.astype(float)

    # Upper anchor: strictly between anchors first
    region = prices_for_upper[anchor1[0]:anchor2[0] + 1]
    if len(region) == 0:
        return None
    max_rel = np.argmax(region)
    high_idx = anchor1[0] + max_rel
    high_price = float(region[max_rel])

    # Fallback: +10 candles right of anchor2
    if high_price <= max(anchor1[1], anchor2[1]):
        ext_start = anchor2[0] + 1
        ext_end = min(n, anchor2[0] + 11)
        region_ext = prices_for_upper[ext_start:ext_end]
        if len(region_ext) > 0:
            ext_rel = np.argmax(region_ext)
            ext_price = float(region_ext[ext_rel])
            if ext_price > high_price:
                high_idx = ext_start + ext_rel
                high_price = ext_price

    upper_int = np.log(high_price) - slope * high_idx

    mid = (anchor1[0] + anchor2[0]) // 2
    um = _price_at(slope, upper_int, mid)
    lm = _price_at(slope, lower_int, mid)
    if um <= lm:
        return None
    width_pct = (um - lm) / lm * 100
    if width_pct < 1.0:
        return None

    lower_touches = _touches(swing_lows_list, slope, lower_int, 0.015)
    swing_highs_for_touch = find_swing_highs_body(df) if use_bodies else find_swing_highs(df)
    upper_touches = _touches(swing_highs_for_touch, slope, upper_int, 0.015)

    last_idx = n - 1
    last_close = float(df["close"].iloc[-1])
    upper_now = _price_at(slope, upper_int, last_idx)
    lower_now = _price_at(slope, lower_int, last_idx)
    position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0

    return {
        "direction": _direction(slope),
        "upper_line": {"slope": slope, "intercept": upper_int, "points": upper_touches},
        "lower_line": {"slope": slope, "intercept": lower_int, "points": lower_touches},
        "width_pct": width_pct,
        "price_position": position,
        "breakout": None,
        "second_channel": None,
        "predicted_channel": None,
        "touches_upper": len(upper_touches),
        "touches_lower": len(lower_touches),
        "swing_highs": find_swing_highs(df),
        "swing_lows": swing_lows_list,
        "anchors": {
            "upper": [(high_idx, high_price)],
            "lower": [anchor1, anchor2],
        },
    }


def _detect_channel_v5(df, interval="4h"):
    """
    Algorithm 5 — Ascending channel from body bottoms.
    Two anchors on body-bottom swing lows: anchor1 (left, LOWER), anchor2 (right, HIGHER).
    Upper anchor = highest body top between the two lower anchors.
    Search window: last 120 candles. Iterates from farthest pair to closest.
    If channel cuts through bodies → try next pair closer to price.
    """
    n = len(df)
    if n < 50:
        return None

    swing_lows_b = find_swing_lows_body(df)
    if len(swing_lows_b) < 2:
        return None

    lookback_start = max(0, n - 120)
    relevant_lows = [(i, p) for i, p in swing_lows_b if i >= lookback_start]
    if len(relevant_lows) < 2:
        return None

    # Iterate from FARTHEST pair to CLOSEST
    # anchor1 = left (lower price), anchor2 = right toward price (higher price)
    for k in range(len(relevant_lows) - 1):
        anchor1 = relevant_lows[k]
        for j in range(k + 1, len(relevant_lows)):
            anchor2 = relevant_lows[j]
            # anchor2 must be HIGHER than anchor1 (ascending)
            if anchor2[1] <= anchor1[1]:
                continue
            ch = _build_channel_from_lows(df, anchor1, anchor2, swing_lows_b, use_bodies=True)
            if ch and _validate_channel_bodies(df, ch):
                return ch
    return None


def _detect_channel_v5_1(df, interval="4h"):
    """
    Algorithm 5.1 — Same as algo 5, but anchors on candle lows (wicks)
    and upper anchor on candle high (wick).
    Fallback when algo 5 doesn't find a valid channel.
    """
    n = len(df)
    if n < 50:
        return None

    swing_lows_reg = find_swing_lows(df)
    if len(swing_lows_reg) < 2:
        return None

    lookback_start = max(0, n - 120)
    relevant_lows = [(i, p) for i, p in swing_lows_reg if i >= lookback_start]
    if len(relevant_lows) < 2:
        return None

    for k in range(len(relevant_lows) - 1):
        anchor1 = relevant_lows[k]
        for j in range(k + 1, len(relevant_lows)):
            anchor2 = relevant_lows[j]
            # anchor2 must be HIGHER (ascending)
            if anchor2[1] <= anchor1[1]:
                continue
            ch = _build_channel_from_lows(df, anchor1, anchor2, swing_lows_reg, use_bodies=False)
            if ch and _validate_channel_bodies(df, ch):
                return ch
    return None


def _detect_channel_v5_2(df, interval="4h"):
    """
    Algorithm 5.2 — Fallback ascending channel.
    Same lower anchors as algo 5 (body bottoms), but if upper anchor between
    anchors causes line to cross bodies, search for upper anchor AFTER anchor2
    (closer to current price) — highest body top in the 10 candles after anchor2.
    """
    n = len(df)
    if n < 50:
        return None

    body_tops = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    body_bottoms = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))

    swing_lows_b = find_swing_lows_body(df)
    if len(swing_lows_b) < 2:
        return None

    lookback_start = max(0, n - 120)
    relevant_lows = [(i, p) for i, p in swing_lows_b if i >= lookback_start]
    if len(relevant_lows) < 2:
        return None

    for k in range(len(relevant_lows) - 1):
        anchor1 = relevant_lows[k]
        for j in range(k + 1, len(relevant_lows)):
            anchor2 = relevant_lows[j]
            if anchor2[1] <= anchor1[1]:
                continue

            slope, lower_int = _log_line(anchor1[0], anchor1[1], anchor2[0], anchor2[1])
            if slope <= 0 or abs(slope) > 0.03:
                continue

            # Search for upper anchor AFTER anchor2 (toward current price)
            ext_start = anchor2[0] + 1
            ext_end = min(n, anchor2[0] + 11)
            if ext_start >= n:
                continue
            region = body_tops[ext_start:ext_end]
            if len(region) == 0:
                continue
            max_rel = np.argmax(region)
            high_idx = ext_start + max_rel
            high_price = float(region[max_rel])

            if high_price <= max(anchor1[1], anchor2[1]):
                continue

            upper_int = np.log(high_price) - slope * high_idx

            mid = (anchor1[0] + anchor2[0]) // 2
            um = _price_at(slope, upper_int, mid)
            lm = _price_at(slope, lower_int, mid)
            if um <= lm:
                continue
            width_pct = (um - lm) / lm * 100
            if width_pct < 1.0:
                continue

            lower_touches = _touches(swing_lows_b, slope, lower_int, 0.015)
            swing_highs_b = find_swing_highs_body(df)
            upper_touches = _touches(swing_highs_b, slope, upper_int, 0.015)

            last_idx = n - 1
            last_close = float(df["close"].iloc[-1])
            upper_now = _price_at(slope, upper_int, last_idx)
            lower_now = _price_at(slope, lower_int, last_idx)
            position = (last_close - lower_now) / (upper_now - lower_now) * 100 if upper_now > lower_now else 50.0

            ch = {
                "direction": _direction(slope),
                "upper_line": {"slope": slope, "intercept": upper_int, "points": upper_touches},
                "lower_line": {"slope": slope, "intercept": lower_int, "points": lower_touches},
                "width_pct": width_pct,
                "price_position": position,
                "breakout": None,
                "second_channel": None,
                "predicted_channel": None,
                "touches_upper": len(upper_touches),
                "touches_lower": len(lower_touches),
                "swing_highs": find_swing_highs(df),
                "swing_lows": swing_lows_b,
                "anchors": {
                    "upper": [(high_idx, high_price)],
                    "lower": [anchor1, anchor2],
                },
            }
            if _validate_channel_bodies(df, ch):
                return ch
    return None


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
    
    # Break detection — by body (not just full candle)
    body_tops_asc = np.maximum(df["open"].values.astype(float), df["close"].values.astype(float))
    body_bots_asc = np.minimum(df["open"].values.astype(float), df["close"].values.astype(float))
    
    breakout = None
    break_idx = None
    
    # Break DOWN: body bottom goes below lower line
    for i in range(anchor2[0] + 1, n):
        lower_at = _price_at(slope, lower_int, i)
        if body_bots_asc[i] < lower_at * 0.997:
            breakout = "down"
            break_idx = i
            break
    
    # Break UP: body top goes above upper line
    if breakout is None:
        for i in range(anchor2[0] + 1, n):
            upper_at = _price_at(slope, upper_int, i)
            if body_tops_asc[i] > upper_at * 1.003:
                breakout = "up"
                break_idx = i
                break
    
    # Filter touch points to only BEFORE break (don't count post-break touches)
    if break_idx is not None:
        lower_touches = [(i, p) for i, p in lower_touches if i <= break_idx]
        upper_touches = [(i, p) for i, p in upper_touches if i <= break_idx]
    
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
        "anchors": {
            "lower": [anchor1, anchor2],
            "upper": [(high_idx, high_price)],
        },
    }
