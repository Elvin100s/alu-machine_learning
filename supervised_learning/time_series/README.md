# Time Series Forecasting — BTC

Forecasting the next hour's BTC-USD close price from the previous 24 hours
of trading, using a recurrent network fed by a `tf.data.Dataset`.

## Data

Minute-level BTC-USD candles from the Coinbase Exchange public API,
covering the two years up to the fetch date. `fetch_data.py` pages the
`/products/BTC-USD/candles` endpoint 300 candles at a time and writes
`btc_1min.csv`.

## Files

| File | Purpose |
| --- | --- |
| `fetch_data.py` | Downloads raw minute candles from Coinbase |
| `preprocess_data.py` | Minute bars to hourly bars, windowing, normalizing, splitting |
| `forecast_btc.py` | `tf.data` input pipeline, LSTM model, training, evaluation |
| `plot_results.py` | Training curve, predicted vs actual, error distribution |

## Pipeline

1. **Aggregate 60s to 1hr.** A minute bar is mostly microstructure noise.
   Hourly bars keep the shape of a trading day and cut the sequence length
   by sixty, so a 24-step window covers a full day rather than 24 minutes.
2. **Repair gaps.** Hours with no trades are carried forward at the last
   close with zero volume, so every window spans exactly 24 hours.
3. **Normalize per window.** BTC's price level is not stationary; a window
   from one year sits nowhere near a window from the next. Each window is
   centred on its own mean and scaled by its own standard deviation, and
   those statistics are kept so predictions convert back to dollars.
4. **Split chronologically.** 70/15/15 by time, never shuffled across the
   boundary — shuffling first would train the model on the future.

## Usage

```
./fetch_data.py                 # writes btc_1min.csv
./preprocess_data.py btc_1min.csv btc_hourly.npz
./forecast_btc.py               # trains, evaluates, saves artifacts
./plot_results.py               # writes the figures
```

## Results

See `RESULTS.md` for the measured metrics, and the accompanying blog post
for the write-up.
