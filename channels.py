"""
Channel detection algorithm.
Works in log-price space for geometrically correct channels on log-scale charts.
"""
import numpy as np
import pandas as pd
from itertools import combinations


def _body_max(df):
    """Max of open/close per candle (body top)."""
    return np.maximum(df["open"].values, df["close"].values)


def _body_min(df):
    """Min of open/close per candle (body bottom)."""
    return np.minimum(df["open"].values, df["close"].values)


def find_swing_highs(df, min_radius=2, max_radius=6, min_distance=3):
    """Find swing highs using flexible radius (like AIAlisa's is_peak_flexible)."""
    prices = _body_max(df)
    n = len(prices)
    candidates = []

    for i in range(max_radius, n - max_radius):
        for r in range(min_radius, max_radius + 1):
            left = prices[max(0, i - r):i]
            right = prices[i + 1:min(n, i + r + 1)]
            if len(left) > 0 and len(right) > 0:
                if prices[i] >= np.max(left) and prices[i] >= np.max(right):
                    candidates.append((i, prices[i]))
                    break

    # Deduplicate nearby points
    if not candidates:
        return []
    candidates.sort(key=lambda x: -x[1])  # highest first for dedup
    result = []
    for idx, price in candidates:
        if all(abs(idx - r[0]) >= min_distance for r in result):
            result.append((idx, price))
    result.sort(key=lambda x: x[0])
    return result


def find_swing_lows(df, min_radius=2, max_radius=6, min_distance=3):
    """Find swing lows using flexible radius."""
    prices = _body_min(df)
    n = len(prices)
    candidates = []

    for i in range(max_radius, n - max_radius):
        for r in range(min_radius, max_radius + 1):
            left = prices[max(0, i - r):i]
            right = prices[i + 1:min(n, i + r + 1)]
            if len(left) > 0 and len(right) > 0:
                if prices[i] <= np.min(left) and prices[i] <= np.min(right):
                    candidates.append((i, prices[i]))
                    break

    if not candidates:
        return []
    candidates.sort(key=lambda x: x[1])  # lowest first for dedup
    result = []
    for idx, price in candidates:
        if all(abs(idx - r[0]) >= min_distance for r in result):
            result.append((idx, price))
    result.sort(key=lambda x: x[0])
    return result


def _log_line(idx_a, price_a, idx_b, price_b):
    """Compute slope and intercept in log space: log(price) = slope * idx + intercept."""
    log_a = np.log(price_a)
    log_b = np.log(price_b)
    slope = (log_b - log_a) / (idx_b - idx_a)
    intercept = log_a - slope * idx_a
    return slope, intercept


def _line_price_at(slope, intercept, idx):
    """Get price at index from log-space line."""
    return np.exp(slope * idx + intercept)


def _count_touches(indices_prices, slope, intercept, tolerance_pct=0.015):
    """Count how many points touch the line within tolerance."""
    touches = []
    for idx, price in indices_prices:
        line_price = _line_price_at(slope, intercept, idx)
        if abs(price - line_price) / line_price <= tolerance_pct:
            touches.append((idx, price))
    return touches


def _count_breaches(df, upper_slope, upper_intercept, lower_slope, lower_intercept):
    """Count candles that breach channel boundaries (using body, not wicks)."""
    body_hi = _body_max(df)
    body_lo = _body_min(df)
    breaches = 0
    n = len(df)
    for i in range(n):
        upper_price = _line_price_at(upper_slope, upper_intercept, i)
        lower_price = _line_price_at(lower_slope, lower_intercept, i)
        if body_hi[i] > upper_price * 1.01:  # 1% tolerance
            breaches += 1
        elif body_lo[i] < lower_price * 0.99:
            breaches += 1
    return breaches


def _build_channel_from_line_pair(df, primary_points, opposite_points,
                                   primary_is_upper, min_span=30):
    """
    Build channel candidates from pairs of primary points.
    primary_is_upper=True: fitting upper line through highs, projecting lower through lows
    primary_is_upper=False: fitting lower line through lows, projecting upper through highs
    """
    n = len(df)
    candidates = []

    for (idx_a, price_a), (idx_b, price_b) in combinations(primary_points, 2):
        span = abs(idx_b - idx_a)
        if span < min_span:
            continue

        slope, intercept = _log_line(idx_a, price_a, idx_b, price_b)

        # Find best opposite point for parallel line
        best_opposite = None
        best_width = 0

        for opp_idx, opp_price in opposite_points:
            # Parallel line: same slope, intercept from this point
            opp_intercept = np.log(opp_price) - slope * opp_idx

            if primary_is_upper:
                upper_slope, upper_intercept = slope, intercept
                lower_slope, lower_intercept = slope, opp_intercept
            else:
                lower_slope, lower_intercept = slope, intercept
                upper_slope, upper_intercept = slope, opp_intercept

            # Check channel makes sense (upper above lower at midpoint)
            mid_idx = (idx_a + idx_b) // 2
            upper_at_mid = _line_price_at(upper_slope, upper_intercept, mid_idx)
            lower_at_mid = _line_price_at(lower_slope, lower_intercept, mid_idx)
            if upper_at_mid <= lower_at_mid:
                continue

            width_pct = (upper_at_mid - lower_at_mid) / lower_at_mid * 100
            if width_pct < 1.0 or width_pct > 100.0:
                continue

            # Count breaches
            breaches = _count_breaches(df, upper_slope, upper_intercept,
                                        lower_slope, lower_intercept)
            breach_pct = breaches / n
            if breach_pct > 0.20:
                continue

            # Count touches on both sides
            if primary_is_upper:
                touches_primary = _count_touches(
                    [(idx_a, price_a), (idx_b, price_b)] +
                    [(i, p) for i, p in primary_points if i != idx_a and i != idx_b],
                    upper_slope, upper_intercept)
                touches_opposite = _count_touches(opposite_points, lower_slope, lower_intercept)
            else:
                touches_primary = _count_touches(
                    [(idx_a, price_a), (idx_b, price_b)] +
                    [(i, p) for i, p in primary_points if i != idx_a and i != idx_b],
                    lower_slope, lower_intercept)
                touches_opposite = _count_touches(opposite_points, upper_slope, upper_intercept)

            if len(touches_primary) < 2 or len(touches_opposite) < 2:
                continue

            if width_pct > best_width or (width_pct > best_width * 0.8 and
                                           len(touches_opposite) > (best_opposite or {}).get('_touches_opp', 0)):
                best_width = width_pct
                best_opposite = {
                    "upper_slope": upper_slope,
                    "upper_intercept": upper_intercept,
                    "lower_slope": lower_slope,
                    "lower_intercept": lower_intercept,
                    "touches_upper": touches_primary if primary_is_upper else touches_opposite,
                    "touches_lower": touches_opposite if primary_is_upper else touches_primary,
                    "width_pct": width_pct,
                    "span": span,
                    "breach_pct": breach_pct,
                    "_touches_opp": len(touches_opposite),
                }

        if best_opposite:
            candidates.append(best_opposite)

    return candidates


def _score_channel(ch, df):
    """Score a channel candidate. Higher is better."""
    n = len(df)
    total_touches = len(ch["touches_upper"]) + len(ch["touches_lower"])
    span_ratio = ch["span"] / n

    # Check if price is currently inside channel
    last_idx = n - 1
    last_close = float(df["close"].iloc[-1])
    upper_now = _line_price_at(ch["upper_slope"], ch["upper_intercept"], last_idx)
    lower_now = _line_price_at(ch["lower_slope"], ch["lower_intercept"], last_idx)

    inside = lower_now <= last_close <= upper_now
    just_broke = (last_close > upper_now and last_close < upper_now * 1.05) or \
                 (last_close < lower_now and last_close > lower_now * 0.95)

    position_bonus = 1.5 if inside else (1.2 if just_broke else 0.5)

    score = total_touches * 2.0 + span_ratio * 10.0 + position_bonus
    score -= ch["breach_pct"] * 5.0

    return score


def _determine_direction(slope):
    """Classify channel direction based on slope magnitude."""
    if abs(slope) < 0.0001:
        return "horizontal"
    return "ascending" if slope > 0 else "descending"


def _compute_breakout(df, channel):
    """Check if price has broken out of channel and compute predicted channel."""
    n = len(df)
    last_idx = n - 1
    last_close = float(df["close"].iloc[-1])

    upper_now = _line_price_at(channel["upper_slope"], channel["upper_intercept"], last_idx)
    lower_now = _line_price_at(channel["lower_slope"], channel["lower_intercept"], last_idx)

    # Channel width in log space
    log_width = channel["upper_intercept"] - channel["lower_intercept"]

    breakout = None
    predicted = None

    if last_close > upper_now * 1.005:
        breakout = "up"
        # Old upper becomes new lower, new upper at width * 0.618 above
        new_lower_intercept = channel["upper_intercept"]
        new_upper_intercept = channel["upper_intercept"] + log_width * 0.618
        predicted = {
            "direction": _determine_direction(channel["upper_slope"]),
            "upper_line": {
                "slope": channel["upper_slope"],
                "intercept": new_upper_intercept,
                "points": [],
            },
            "lower_line": {
                "slope": channel["upper_slope"],
                "intercept": new_lower_intercept,
                "points": [],
            },
        }
    elif last_close < lower_now * 0.995:
        breakout = "down"
        # Old lower becomes new upper, new lower at width below
        new_upper_intercept = channel["lower_intercept"]
        new_lower_intercept = channel["lower_intercept"] - log_width
        predicted = {
            "direction": _determine_direction(channel["lower_slope"]),
            "upper_line": {
                "slope": channel["lower_slope"],
                "intercept": new_upper_intercept,
                "points": [],
            },
            "lower_line": {
                "slope": channel["lower_slope"],
                "intercept": new_lower_intercept,
                "points": [],
            },
        }

    return breakout, predicted


def detect_channel(df):
    """
    Main entry point. Detect the best price channel from OHLCV data.
    Returns channel dict or None.
    """
    if len(df) < 50:
        return None

    swing_highs = find_swing_highs(df)
    swing_lows = find_swing_lows(df)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    # Build candidates from both directions
    candidates = []

    # Descending/horizontal channels: fit upper line through highs
    candidates.extend(
        _build_channel_from_line_pair(df, swing_highs, swing_lows, primary_is_upper=True)
    )

    # Ascending/horizontal channels: fit lower line through lows
    candidates.extend(
        _build_channel_from_line_pair(df, swing_lows, swing_highs, primary_is_upper=False)
    )

    if not candidates:
        return None

    # Score and pick best
    best = max(candidates, key=lambda ch: _score_channel(ch, df))

    # Compute current position
    n = len(df)
    last_idx = n - 1
    last_close = float(df["close"].iloc[-1])
    upper_now = _line_price_at(best["upper_slope"], best["upper_intercept"], last_idx)
    lower_now = _line_price_at(best["lower_slope"], best["lower_intercept"], last_idx)

    if upper_now > lower_now:
        price_position = (last_close - lower_now) / (upper_now - lower_now) * 100
        price_position = max(0, min(200, price_position))  # allow >100 for breakout display
    else:
        price_position = 50.0

    direction = _determine_direction(best["upper_slope"])
    breakout, predicted = _compute_breakout(df, best)

    return {
        "direction": direction,
        "upper_line": {
            "slope": best["upper_slope"],
            "intercept": best["upper_intercept"],
            "points": [(idx, price) for idx, price in best["touches_upper"]],
        },
        "lower_line": {
            "slope": best["lower_slope"],
            "intercept": best["lower_intercept"],
            "points": [(idx, price) for idx, price in best["touches_lower"]],
        },
        "width_pct": best["width_pct"],
        "price_position": price_position,
        "breakout": breakout,
        "predicted_channel": predicted,
        "touches_upper": len(best["touches_upper"]),
        "touches_lower": len(best["touches_lower"]),
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
    }
