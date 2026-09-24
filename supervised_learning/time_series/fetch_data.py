#!/usr/bin/env python3
"""Fetches BTC-USD minute candles from Coinbase using a small thread pool."""
import csv
import datetime
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = ('https://api.exchange.coinbase.com/products/BTC-USD/candles'
       '?granularity=60&start={start}&end={end}')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'btc_1min.csv')
DAYS, WORKERS = 730, 6
CHUNK = datetime.timedelta(minutes=300)
rows, lock, done = {}, threading.Lock(), [0]


def iso(dt):
    """Formats a datetime as ISO-8601."""
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def grab(span):
    """Fetches one chunk of candles and merges it into the shared table."""
    start, end = span
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                URL.format(start=iso(start), end=iso(end)),
                headers={'User-Agent': 'curl'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            with lock:
                for candle in data:
                    rows[int(candle[0])] = candle
                done[0] += 1
                if done[0] % 200 == 0:
                    print('%d chunks, %d candles' % (done[0], len(rows)),
                          flush=True)
            return
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    with lock:
        done[0] += 1


def main():
    """Fetches the full range and writes it to a CSV."""
    end_all = datetime.datetime.now(datetime.timezone.utc).replace(
        tzinfo=None, second=0, microsecond=0)
    start_all = end_all - datetime.timedelta(days=DAYS)
    spans, cur = [], start_all
    while cur < end_all:
        nxt = min(cur + CHUNK, end_all)
        spans.append((cur, nxt))
        cur = nxt
    print('%d chunks with %d workers' % (len(spans), WORKERS), flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(grab, spans))
    with open(OUT, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time', 'low', 'high', 'open', 'close', 'volume'])
        for stamp in sorted(rows):
            writer.writerow(rows[stamp])
    print('DONE %d candles -> %s' % (len(rows), OUT), flush=True)


if __name__ == '__main__':
    main()
