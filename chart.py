"""
Chart drawing for channel detection bot.
Style inspired by AIAlisa: mplfinance, charles style, log scale, RSI panel.
"""
import math
import os
import gc
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.ticker import ScalarFormatter, FuncFormatter


def _fmt_price(price: float) -> str:
    """Smart price formatting: up to 8 decimals, trim trailing zeros keeping 1."""
    s = f"{price:.8f}"
    int_part, dec_part = s.split('.')
    stripped = dec_part.rstrip('0')
    if len(stripped) == 0:
        return f"{int_part}.00"
    keep = min(len(stripped) + 1, 8)
    keep = max(keep, 2)
    return f"{int_part}.{dec_part[:keep]}"


def _compute_rsi(close, period):
    """Compute RSI using exponential moving average."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    alpha = 1.0 / period
    avg_g = np.zeros(len(gain))
    avg_l = np.zeros(len(gain))
    avg_g[0] = gain.iloc[0] if len(gain) > 0 else 0
    avg_l[0] = loss.iloc[0] if len(loss) > 0 else 0
    for i in range(1, len(gain)):
        avg_g[i] = alpha * gain.iloc[i] + (1 - alpha) * avg_g[i - 1]
        avg_l[i] = alpha * loss.iloc[i] + (1 - alpha) * avg_l[i - 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_l == 0, 100, avg_g / avg_l)
    return pd.Series(100 - (100 / (1 + rs)), index=close.index)


def _apply_custom_grid(ax, plot_df, view_limit):
    """Custom grid: vertical every 20 candles, horizontal every 10% from current price."""
    ax.grid(False)
    ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax.tick_params(axis='y', which='both', labelright=False, labelleft=False, right=False, left=False)
    ax.set_ylabel('')

    for i in range(0, view_limit, 20):
        ax.axvline(x=i, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)

    y_low, y_high = ax.get_ylim()
    current_price = float(plot_df['close'].iloc[-1])
    if current_price > 0 and y_low > 0 and y_high > 0:
        n = 1
        while True:
            level = current_price * (1 + n * 0.10)
            if level > y_high:
                break
            ax.axhline(y=level, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)
            n += 1
        n = 1
        while True:
            level = current_price * (1 - n * 0.10)
            if level <= 0 or level < y_low:
                break
            ax.axhline(y=level, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)
            n += 1


def _apply_date_labels(ax_main, fig, plot_df, view_limit, axlist=None):
    """Draw date labels between main chart and RSI panel."""
    fig.canvas.draw()
    bbox_main = ax_main.get_position()
    main_bottom = bbox_main.y0

    rsi_top = main_bottom - 0.05
    if axlist is not None:
        panels = {}
        for a in axlist:
            pnum = getattr(a, '_panel_num', None)
            if pnum is not None and pnum not in panels:
                panels[pnum] = a
        if not panels.get(1) and len(axlist) >= 3:
            panels[1] = axlist[1]
        if panels.get(1):
            rsi_top = panels[1].get_position().y1

    y_fig = (main_bottom + rsi_top) / 2

    for i in range(0, view_limit, 20):
        if i < len(plot_df):
            dt = plot_df.index[i]
            label = f"{dt.strftime('%b %d')}\n{dt.strftime('%H:%M')}"
            disp = ax_main.transData.transform((i, 0))
            fig_x = fig.transFigure.inverted().transform(disp)[0]
            fig.text(fig_x, y_fig, label,
                     color='black', fontsize=7, fontweight='bold',
                     ha='center', va='center')


def _line_price_at(slope, intercept, idx):
    """Get price at index from log-space line."""
    return np.exp(slope * idx + intercept)


def draw_channel_chart(symbol, df, channel, interval="4h"):
    """
    Draw the channel chart and save as PNG.
    Returns file path.
    """
    view_limit = min(len(df), 200)
    plot_df = df.iloc[-view_limit:].copy().reset_index(drop=True)
    plot_df['ds'] = pd.to_datetime(plot_df['open_time'], unit='ms')
    plot_df.set_index('ds', inplace=True)

    close = plot_df['close'].astype(float)

    # RSI
    rsi6 = _compute_rsi(close, 6)
    rsi12 = _compute_rsi(close, 12)
    rsi24 = _compute_rsi(close, 24)

    rsi_addplots = [
        mpf.make_addplot(rsi6, panel=1, color='#F0B90B', width=1.2, ylabel='RSI'),
        mpf.make_addplot(rsi12, panel=1, color='#E040FB', width=1.0),
        mpf.make_addplot(rsi24, panel=1, color='#7B1FA2', width=1.0),
    ]

    # Channel lines as addplots
    extra_addplots = []

    # Precompute safe price bounds for clamping lines
    _price_floor = float(plot_df['low'].min()) * 0.5
    _price_ceil = float(plot_df['high'].max()) * 2.0

    if channel:
        upper_vals = []
        lower_vals = []
        mid_vals = []
        for i in range(view_limit):
            u = _line_price_at(channel["upper_line"]["slope"],
                               channel["upper_line"]["intercept"], i)
            l = _line_price_at(channel["lower_line"]["slope"],
                               channel["lower_line"]["intercept"], i)
            u = max(min(u, _price_ceil), _price_floor)
            l = max(min(l, _price_ceil), _price_floor)
            upper_vals.append(u)
            lower_vals.append(l)
            mid_vals.append(np.exp((np.log(u) + np.log(l)) / 2))

        extra_addplots.append(mpf.make_addplot(upper_vals, color='cyan', width=2, linestyle='-'))
        extra_addplots.append(mpf.make_addplot(lower_vals, color='cyan', width=2, linestyle='-'))
        extra_addplots.append(mpf.make_addplot(mid_vals, color='cyan', width=0.8, linestyle='--'))

        # Swing point markers
        swing_hi_series = [float('nan')] * view_limit
        swing_lo_series = [float('nan')] * view_limit
        for idx, price in channel.get("swing_highs", []):
            if 0 <= idx < view_limit:
                swing_hi_series[idx] = price
        for idx, price in channel.get("swing_lows", []):
            if 0 <= idx < view_limit:
                swing_lo_series[idx] = price

        if not all(np.isnan(x) for x in swing_hi_series):
            extra_addplots.append(mpf.make_addplot(
                swing_hi_series, type='scatter', markersize=40,
                marker='v', color='red', alpha=0.8))
        if not all(np.isnan(x) for x in swing_lo_series):
            extra_addplots.append(mpf.make_addplot(
                swing_lo_series, type='scatter', markersize=40,
                marker='^', color='lime', alpha=0.8))

        # Second channel (post-breakout) — orange lines
        ch2 = channel.get("second_channel")
        if ch2:
            ch2_upper = []
            ch2_lower = []
            ch2_mid = []
            for i in range(view_limit):
                pu = _line_price_at(ch2["upper_line"]["slope"],
                                    ch2["upper_line"]["intercept"], i)
                pl = _line_price_at(ch2["lower_line"]["slope"],
                                    ch2["lower_line"]["intercept"], i)
                pu = max(min(pu, _price_ceil), _price_floor)
                pl = max(min(pl, _price_ceil), _price_floor)
                ch2_upper.append(pu)
                ch2_lower.append(pl)
                ch2_mid.append(np.exp((np.log(pu) + np.log(pl)) / 2))
            extra_addplots.append(mpf.make_addplot(
                ch2_upper, color='#FFA500', width=2, linestyle='-'))
            extra_addplots.append(mpf.make_addplot(
                ch2_lower, color='#FFA500', width=2, linestyle='-'))
            extra_addplots.append(mpf.make_addplot(
                ch2_mid, color='#FFA500', width=0.8, linestyle='--'))

    # Title
    direction_str = channel["direction"].title() if channel else "No"
    tf_label = interval.upper()
    title = f"\n{symbol}USDT {tf_label} | {direction_str} Channel"

    fig = None
    file_path = f"channel_{symbol}_{interval}.png"

    try:
        fig, axlist = mpf.plot(
            plot_df, type='candle', style='charles',
            addplot=extra_addplots + rsi_addplots,
            yscale='log',
            title=title,
            figsize=(14, 10),
            returnfig=True,
            tight_layout=False,
            panel_ratios=(10, 2),
        )

        ax = axlist[0]
        ax.set_xlim(-0.5, view_limit - 0.5)

        # Tight Y-axis with log padding
        candle_low = float(plot_df['low'].min())
        candle_high = float(plot_df['high'].max())

        # Extend ylim slightly for predicted channel but cap at 30% beyond candles
        if channel and channel.get("predicted_channel"):
            pred = channel["predicted_channel"]
            # Only check at the right edge (current candle area)
            last_q = view_limit - 1
            pu = _line_price_at(pred["upper_line"]["slope"],
                                pred["upper_line"]["intercept"], last_q)
            pl = _line_price_at(pred["lower_line"]["slope"],
                                pred["lower_line"]["intercept"], last_q)
            # Cap extension at 30% beyond candle range
            max_extend_hi = candle_high * 1.30
            min_extend_lo = candle_low * 0.70
            candle_high = min(max(candle_high, pu), max_extend_hi)
            candle_low = max(min(candle_low, pl), min_extend_lo)

        if candle_low > 0 and candle_high > candle_low:
            log_lo = math.log(candle_low)
            log_hi = math.log(candle_high)
            log_pad = (log_hi - log_lo) * 0.05
            ax.set_ylim(math.exp(log_lo - log_pad), math.exp(log_hi + log_pad))

        # Channel fill (semi-transparent)
        if channel:
            upper_fill = [_line_price_at(channel["upper_line"]["slope"],
                                          channel["upper_line"]["intercept"], i)
                          for i in range(view_limit)]
            lower_fill = [_line_price_at(channel["lower_line"]["slope"],
                                          channel["lower_line"]["intercept"], i)
                          for i in range(view_limit)]
            xs = list(range(view_limit))
            ax.fill_between(xs, lower_fill, upper_fill,
                            color='cyan', alpha=0.08, zorder=1)

            # Second channel fill (orange)
            ch2 = channel.get("second_channel")
            if ch2:
                ch2_upper_fill = [_line_price_at(ch2["upper_line"]["slope"],
                                                  ch2["upper_line"]["intercept"], i)
                                  for i in range(view_limit)]
                ch2_lower_fill = [_line_price_at(ch2["lower_line"]["slope"],
                                                  ch2["lower_line"]["intercept"], i)
                                  for i in range(view_limit)]
                ax.fill_between(xs, ch2_lower_fill, ch2_upper_fill,
                                color='#FFA500', alpha=0.06, zorder=1)
                # Touch points on second channel
                for idx, price in ch2["upper_line"]["points"]:
                    if 0 <= idx < view_limit:
                        ax.plot(idx, price, 'o', color='#FFA500', markersize=6, zorder=5)
                for idx, price in ch2["lower_line"]["points"]:
                    if 0 <= idx < view_limit:
                        ax.plot(idx, price, 'o', color='#FFA500', markersize=6, zorder=5)

            # Touch point labels on channel lines
            for idx, price in channel["upper_line"]["points"]:
                if 0 <= idx < view_limit:
                    ax.plot(idx, price, 'o', color='cyan', markersize=6, zorder=5)
            for idx, price in channel["lower_line"]["points"]:
                if 0 <= idx < view_limit:
                    ax.plot(idx, price, 'o', color='cyan', markersize=6, zorder=5)

            # ANCHOR POINTS — big markers showing where lines are built from
            anchors = channel.get("anchors")
            if anchors:
                for idx, price in anchors.get("upper", []):
                    if 0 <= idx < view_limit:
                        ax.plot(idx, price, 'D', color='#FFD700', markersize=12,
                                markeredgecolor='white', markeredgewidth=1.5, zorder=8)
                for idx, price in anchors.get("lower", []):
                    if 0 <= idx < view_limit:
                        ax.plot(idx, price, 'D', color='#FFD700', markersize=12,
                                markeredgecolor='white', markeredgewidth=1.5, zorder=8)
            ch2 = channel.get("second_channel")
            if ch2:
                ch2_anchors = ch2.get("anchors")
                if ch2_anchors:
                    for idx, price in ch2_anchors.get("upper", []):
                        if 0 <= idx < view_limit:
                            ax.plot(idx, price, 'D', color='#FFD700', markersize=12,
                                    markeredgecolor='white', markeredgewidth=1.5, zorder=8)
                    for idx, price in ch2_anchors.get("lower", []):
                        if 0 <= idx < view_limit:
                            ax.plot(idx, price, 'D', color='#FFD700', markersize=12,
                                    markeredgecolor='white', markeredgewidth=1.5, zorder=8)

            # Info box removed — was cluttering the chart

            # EXTRA CHANNELS from other algorithms (different colors)
            _extra_colors = ['#00FF00', '#FF69B4', '#FFD700']  # green, pink, gold
            for ec_idx, ec in enumerate(channel.get("extra_channels", [])):
                ec_color = _extra_colors[ec_idx % len(_extra_colors)]
                ec_upper = [_line_price_at(ec["upper_line"]["slope"],
                                           ec["upper_line"]["intercept"], i)
                            for i in range(view_limit)]
                ec_lower = [_line_price_at(ec["lower_line"]["slope"],
                                           ec["lower_line"]["intercept"], i)
                            for i in range(view_limit)]
                ax.plot(range(view_limit), ec_upper, color=ec_color, linewidth=1.5,
                        linestyle='--', alpha=0.7, zorder=3)
                ax.plot(range(view_limit), ec_lower, color=ec_color, linewidth=1.5,
                        linestyle='--', alpha=0.7, zorder=3)
                ax.fill_between(range(view_limit), ec_lower, ec_upper,
                                color=ec_color, alpha=0.04, zorder=1)
                # Extra channel anchors (smaller diamonds)
                ec_anchors = ec.get("anchors", {})
                for label_pts in [ec_anchors.get("upper", []), ec_anchors.get("lower", [])]:
                    for idx, price in label_pts:
                        if 0 <= idx < view_limit:
                            ax.plot(idx, price, 'D', color=ec_color, markersize=8,
                                    markeredgecolor='white', markeredgewidth=1, zorder=7)

        # Watermark
        wm_x = 1 / max(view_limit - 1, 1)
        ax.text(wm_x, 0.02, 'Channel Bot', transform=ax.transAxes,
                color='black', fontsize=24, fontweight='bold',
                ha='left', va='bottom', alpha=0.15)

        # Custom grid
        _apply_custom_grid(ax, plot_df, view_limit)

        # Style RSI panel
        panels = {}
        for a in axlist:
            pnum = getattr(a, '_panel_num', None)
            if pnum is not None and pnum not in panels:
                panels[pnum] = a
        if not panels.get(1) and len(axlist) >= 3:
            panels[1] = axlist[1]
        elif not panels.get(1) and len(axlist) >= 2:
            panels[1] = axlist[1]

        ax_rsi = panels.get(1)
        if ax_rsi:
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(False)
            ax_rsi.tick_params(axis='both', labelsize=5, colors='black')
            ax_rsi.tick_params(axis='x', labelbottom=False)

            for level in [20, 40, 60, 80, 100]:
                ax_rsi.axhline(y=level, color='#404040', linewidth=0.5, alpha=0.4, zorder=0)
            ax_rsi.axhline(y=70, color='#F23645', linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
            ax_rsi.axhline(y=30, color='#089981', linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)

            r6 = float(rsi6.iloc[-1]) if len(rsi6) > 0 else 0
            r12 = float(rsi12.iloc[-1]) if len(rsi12) > 0 else 0
            r24 = float(rsi24.iloc[-1]) if len(rsi24) > 0 else 0
            ax_rsi.text(0.01, 0.95, f"RSI(6): {r6:.2f}", color='#F0B90B',
                        fontsize=5.5, fontweight='bold', transform=ax_rsi.transAxes,
                        va='top', ha='left')
            ax_rsi.text(0.18, 0.95, f"RSI(12): {r12:.2f}", color='#E040FB',
                        fontsize=5.5, fontweight='bold', transform=ax_rsi.transAxes,
                        va='top', ha='left')
            ax_rsi.text(0.38, 0.95, f"RSI(24): {r24:.2f}", color='#7B1FA2',
                        fontsize=5.5, fontweight='bold', transform=ax_rsi.transAxes,
                        va='top', ha='left')

        # No manual layout — let mplfinance handle panel positioning
        # panel 0 = candles (top), panel 1 = RSI (bottom)

        # Date labels
        _apply_date_labels(ax, fig, plot_df, view_limit, axlist=axlist)

        # Nuke offset text
        for _ax in axlist:
            fmt = _ax.yaxis.get_major_formatter()
            if isinstance(fmt, ScalarFormatter):
                fmt.set_useOffset(False)
            ot = _ax.yaxis.get_offset_text()
            ot.set_visible(False)
            ot.set_text("")

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1)

    except Exception as e:
        import traceback
        traceback.print_exc()
        out_path = file_path
    finally:
        if fig:
            fig.clf()
        plt.close('all')
        gc.collect()

    return out_path
