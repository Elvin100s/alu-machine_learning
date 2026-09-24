#!/usr/bin/env python3
"""Checks the hourly BTC series for stationarity and repeating structure.

Two questions decide how the data has to be prepared. Is the series
stationary, meaning its statistical behaviour stays put over time? And
does it carry a repeating pattern tied to the clock or the calendar? The
answers here are what justify the preprocessing choices.
"""
import datetime
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

matplotlib.use('Agg')

INK = '#1c1c1c'
ACCENT = '#c2410c'
MUTED = '#8a8a8a'


def hourly_from_npz(path):
    """Rebuilds a continuous hourly close series from the saved splits.

    Args:
        path: path to the .npz written by preprocess_data.py

    Returns:
        t: numpy.ndarray of timestamps
        close: numpy.ndarray of close prices in USD
    """
    raw = np.load(path)
    t, close = [], []
    for name in ('train', 'val', 'test'):
        stats, y = raw[name + '_stats'], raw[name + '_y']
        close.append(y[:, 0] * stats[:, 1] + stats[:, 0])
        t.append(raw[name + '_t'])
    return np.concatenate(t), np.concatenate(close)


def stationarity(close):
    """Runs augmented Dickey-Fuller on the price and on its log returns.

    The null hypothesis is that the series has a unit root, i.e. that it
    wanders without returning to a stable level. A high p-value means the
    null survives and the series is not stationary.

    Args:
        close: numpy.ndarray of close prices

    Returns:
        a dict with the statistic and p-value for both series
    """
    returns = np.diff(np.log(close))
    price_test = adfuller(close, autolag='AIC')
    return_test = adfuller(returns, autolag='AIC')
    return {
        'price': {'adf': float(price_test[0]), 'p': float(price_test[1])},
        'log_returns': {'adf': float(return_test[0]),
                        'p': float(return_test[1])},
        'price_mean_first_half': float(close[:len(close) // 2].mean()),
        'price_mean_second_half': float(close[len(close) // 2:].mean()),
        'return_std_first_half': float(returns[:len(returns) // 2].std()),
        'return_std_second_half': float(returns[len(returns) // 2:].std())
    }


def seasonality(t, close):
    """Measures average absolute return by hour of day and by weekday.

    Crypto trades continuously, so there is no opening bell to create the
    intraday shape equity markets have. Whatever pattern exists comes from
    when humans are awake, and this is what measures it.

    Args:
        t: numpy.ndarray of timestamps
        close: numpy.ndarray of close prices

    Returns:
        a dict of the per-hour and per-weekday profiles, in basis points
    """
    returns = np.abs(np.diff(np.log(close))) * 10000
    stamps = [datetime.datetime.utcfromtimestamp(v) for v in t[1:]]
    hours = np.array([s.hour for s in stamps])
    days = np.array([s.weekday() for s in stamps])
    by_hour = [float(returns[hours == h].mean()) for h in range(24)]
    by_day = [float(returns[days == d].mean()) for d in range(7)]
    return {'by_hour_bp': by_hour, 'by_weekday_bp': by_day,
            'hour_spread_pct': float(
                (max(by_hour) - min(by_hour)) / np.mean(by_hour) * 100)}


def style(ax, title, xlabel, ylabel):
    """Applies a consistent look to an axis."""
    ax.set_title(title, color=INK, fontsize=12, pad=10, loc='left')
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#d9d9d9')
    ax.grid(True, color='#ececec', linewidth=0.8)
    ax.set_axisbelow(True)


def plot(t, close, seasons, out):
    """Draws the price history and the intraday volatility profile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
    stamps = [datetime.datetime.utcfromtimestamp(v) for v in t]
    axes[0].plot(stamps, close, color=INK, linewidth=1.2)
    style(axes[0], 'BTC-USD hourly close', 'date (UTC)', 'USD')
    axes[1].bar(range(24), seasons['by_hour_bp'], color=ACCENT, alpha=0.85)
    style(axes[1], 'Mean absolute hourly return by hour of day',
          'hour (UTC)', 'basis points')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close(fig)


def main(data='btc_hourly.npz', out_json='analysis.json',
         out_fig='fig_series.png'):
    """Runs both analyses and saves the numbers and the figure."""
    t, close = hourly_from_npz(data)
    report = {'hours': int(len(close)),
              'first': datetime.datetime.utcfromtimestamp(
                  t[0]).strftime('%Y-%m-%d'),
              'last': datetime.datetime.utcfromtimestamp(
                  t[-1]).strftime('%Y-%m-%d'),
              'stationarity': stationarity(close)}
    report['seasonality'] = seasonality(t, close)
    plot(t, close, report['seasonality'], out_fig)
    with open(out_json, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
