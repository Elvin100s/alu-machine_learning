# Results

Data: 17,497 hourly bars, 2024-09-25 to 2026-09-24, aggregated from 1,050,295 minute candles.
Split chronologically 70/15/15: 12,247 train / 2,625 val / 2,625 test windows.

## Stationarity

| Series | ADF statistic | p-value | Verdict |
| --- | --- | --- | --- |
| Hourly close | -1.763 | 0.399 | non-stationary |
| Log returns | -134.2 | < 1e-10 | stationary |

Mean price moved from $96,732 in the first half of the window to $80,309 in the second.
Return volatility barely moved: 0.00493 to 0.00471.

## Weekly cycle

Mean absolute hourly return, basis points:

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
| --- | --- | --- | --- | --- | --- | --- |
| 38.2 | 35.4 | 34.0 | 34.6 | 35.5 | 17.7 | 24.0 |

Hour-of-day spread across the 24 hours: 90% of the mean.

## Model

LSTM(64) -> Dropout(0.2) -> Dense(1), Adam at 1e-3, MSE loss, batch 256.
Early stopping on validation loss, patience 8. Best epoch: 12 of 20 run.
Best validation loss: 0.6357 (normalized units).

## Test performance (2,625 hours held out)

| Metric | LSTM | Persistence baseline |
| --- | --- | --- |
| MAE | $180.73 | $179.21 |
| RMSE | $285.47 | $284.18 |
| Directional accuracy | 48.0% | - |

The LSTM is 0.8% worse than persistence on MAE and calls direction
slightly worse than a coin flip. See `fig_zoom.png`: the model output sits
almost exactly on the baseline.

## Figures

| File | Shows |
| --- | --- |
| `fig_series.png` | Two years of hourly close, and volatility by hour of day |
| `fig_training_loss.png` | Training and validation loss per epoch |
| `fig_predictions.png` | Predicted vs actual over the last 336 test hours |
| `fig_zoom.png` | The same forecast at 48-hour zoom, with the baseline |
| `fig_errors.png` | Error distribution against the baseline |
