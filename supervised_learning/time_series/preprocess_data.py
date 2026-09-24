#!/usr/bin/env python3
"""Preprocesses raw minute-level BTC-USD candles into hourly training data.

The raw feed is one candle per minute. Minute bars are noisy and mostly
report microstructure rather than direction, so they are aggregated into
hourly bars before any model sees them. The hourly series is then cut into
sliding windows of 24 hours, each paired with the close of the hour that
follows it.
"""
import csv
import numpy as np


RAW_COLUMNS = ['time', 'low', 'high', 'open', 'close', 'volume']
HOUR = 3600
WINDOW = 24


def load_raw(path):
    """Loads the raw minute candles from a CSV file.

    Args:
        path: path to a CSV with the columns in RAW_COLUMNS

    Returns:
        a numpy.ndarray of shape (n, 6) sorted by timestamp
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != RAW_COLUMNS:
            raise ValueError('unexpected columns: {}'.format(header))
        for row in reader:
            rows.append([float(v) for v in row])
    data = np.array(rows, dtype=np.float64)
    return data[np.argsort(data[:, 0])]


def to_hourly(minutes):
    """Aggregates minute candles into hourly candles.

    Open is the first price of the hour, close the last, high and low the
    extremes, and volume the sum. Hours with no trades at all are dropped
    here and repaired by fill_gaps.

    Args:
        minutes: numpy.ndarray of shape (n, 6) of minute candles

    Returns:
        numpy.ndarray of shape (h, 3) of [timestamp, close, volume]
    """
    bucket = (minutes[:, 0] // HOUR).astype(np.int64)
    edges = np.flatnonzero(np.diff(bucket)) + 1
    groups = np.split(np.arange(len(minutes)), edges)
    out = np.empty((len(groups), 3), dtype=np.float64)
    for i, idx in enumerate(groups):
        out[i, 0] = bucket[idx[0]] * HOUR
        out[i, 1] = minutes[idx[-1], 4]
        out[i, 2] = minutes[idx, 5].sum()
    return out


def fill_gaps(hourly):
    """Inserts any missing hours so the series is evenly spaced.

    A gap means the exchange reported no trades, not that time stopped, so
    the close is carried forward and the volume recorded as zero. An RNN
    reading a fixed 24-step window assumes a constant step; silently
    skipping hours would make some windows span far more than a day.

    Args:
        hourly: numpy.ndarray of shape (h, 3)

    Returns:
        numpy.ndarray of shape (h_full, 3) with no missing hours
    """
    start, end = int(hourly[0, 0]), int(hourly[-1, 0])
    stamps = np.arange(start, end + HOUR, HOUR, dtype=np.float64)
    out = np.empty((len(stamps), 3), dtype=np.float64)
    out[:, 0] = stamps
    lookup = {int(t): i for i, t in enumerate(hourly[:, 0])}
    last = hourly[0, 1]
    for i, t in enumerate(stamps):
        j = lookup.get(int(t))
        if j is None:
            out[i, 1], out[i, 2] = last, 0.0
        else:
            out[i, 1], out[i, 2] = hourly[j, 1], hourly[j, 2]
            last = hourly[j, 1]
    return out


def make_windows(hourly, window=WINDOW):
    """Cuts the hourly series into sliding windows and their next-hour close.

    Each window is normalized by its own mean and standard deviation. The
    price level of BTC is not stationary - a window from one year sits at a
    completely different level than a window from the next - but the shape
    of a day of trading is comparable across time. Normalizing per window
    removes the level and leaves the shape, and the statistics are kept so
    predictions can be returned to dollars.

    Args:
        hourly: numpy.ndarray of shape (h, 3) of [timestamp, close, volume]
        window: number of hours per input window

    Returns:
        x: numpy.ndarray of shape (n, window, 2)
        y: numpy.ndarray of shape (n, 1), the normalized next-hour close
        stats: numpy.ndarray of shape (n, 2) of the per-window mean and std
        t: numpy.ndarray of shape (n,), the timestamp being predicted
    """
    close, volume = hourly[:, 1], hourly[:, 2]
    log_volume = np.log1p(volume)
    n = len(hourly) - window
    x = np.empty((n, window, 2), dtype=np.float32)
    y = np.empty((n, 1), dtype=np.float32)
    stats = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        chunk = close[i:i + window]
        mean, std = chunk.mean(), chunk.std()
        if std < 1e-8:
            std = 1e-8
        x[i, :, 0] = (chunk - mean) / std
        x[i, :, 1] = log_volume[i:i + window]
        y[i, 0] = (close[i + window] - mean) / std
        stats[i] = (mean, std)
    vol_mean, vol_std = x[:, :, 1].mean(), x[:, :, 1].std()
    x[:, :, 1] = (x[:, :, 1] - vol_mean) / max(vol_std, 1e-8)
    return x, y, stats, hourly[window:, 0]


def split(x, y, stats, t, train=0.7, val=0.15):
    """Splits the windows chronologically into train, validation and test.

    The split is by time, never shuffled, because shuffling first would let
    the model train on hours that come after the ones it is tested on.

    Args:
        x, y, stats, t: arrays returned by make_windows
        train: fraction of windows used for training
        val: fraction of windows used for validation

    Returns:
        a dict of (x, y, stats, t) tuples keyed by 'train', 'val' and 'test'
    """
    n = len(x)
    a, b = int(n * train), int(n * (train + val))
    parts = {}
    for name, lo, hi in (('train', 0, a), ('val', a, b), ('test', b, n)):
        parts[name] = (x[lo:hi], y[lo:hi], stats[lo:hi], t[lo:hi])
    return parts


def preprocess(raw_path, out_path):
    """Runs the whole preprocessing pipeline and saves the result.

    Args:
        raw_path: path to the raw minute-level CSV
        out_path: path of the .npz file to write

    Returns:
        the dict of splits that was saved
    """
    minutes = load_raw(raw_path)
    hourly = fill_gaps(to_hourly(minutes))
    x, y, stats, t = make_windows(hourly)
    parts = split(x, y, stats, t)
    saved = {}
    for name, (xx, yy, ss, tt) in parts.items():
        saved[name + '_x'] = xx
        saved[name + '_y'] = yy
        saved[name + '_stats'] = ss
        saved[name + '_t'] = tt
    np.savez_compressed(out_path, **saved)
    return parts


if __name__ == '__main__':
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else 'btc_1min.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'btc_hourly.npz'
    splits = preprocess(raw, out)
    for key in ('train', 'val', 'test'):
        print('{:5s} x={} y={}'.format(key, splits[key][0].shape,
                                       splits[key][1].shape))
