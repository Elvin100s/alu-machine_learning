#!/usr/bin/env python3
"""Trains an RNN to forecast the next hour's BTC-USD close price.

The model reads the previous 24 hours of normalized close prices and
volume and predicts the close of the following hour. Data is fed through
a tf.data.Dataset so the windows are shuffled, batched and prefetched
without ever materialising a second copy of the training set.
"""
import json
import numpy as np
import tensorflow as tf


BATCH = 256
SHUFFLE_BUFFER = 8192
EPOCHS = 60
PATIENCE = 8


def load_splits(path):
    """Loads the preprocessed splits saved by preprocess_data.py.

    Args:
        path: path to the .npz file

    Returns:
        a dict of (x, y, stats, t) tuples keyed by split name
    """
    raw = np.load(path)
    return {name: (raw[name + '_x'], raw[name + '_y'],
                   raw[name + '_stats'], raw[name + '_t'])
            for name in ('train', 'val', 'test')}


def make_dataset(x, y, training=False, batch=BATCH):
    """Wraps a split in a tf.data.Dataset.

    Training data is shuffled every epoch so consecutive hours do not
    arrive together in a batch; validation and test keep their order.
    cache keeps the windows in memory after the first pass and prefetch
    overlaps batch preparation with the step before it.

    Args:
        x: numpy.ndarray of shape (n, window, features)
        y: numpy.ndarray of shape (n, 1)
        training: whether to shuffle and repeat
        batch: batch size

    Returns:
        a tf.data.Dataset yielding (inputs, target) pairs
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.cache()
    if training:
        ds = ds.shuffle(min(SHUFFLE_BUFFER, len(x)),
                        reshuffle_each_iteration=True)
    ds = ds.batch(batch)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(window, features):
    """Builds the forecasting network.

    A single LSTM layer is enough here. The input is 24 steps long, so
    there is little long-range structure for a deeper stack to find, and
    the dataset is small enough that extra capacity mostly buys
    overfitting. Dropout sits between the recurrent layer and the output.

    Args:
        window: number of timesteps per input
        features: number of features per timestep

    Returns:
        a compiled tf.keras.Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, features)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse', metrics=['mae'])
    return model


def denormalize(values, stats):
    """Converts normalized closes back into dollars.

    Args:
        values: numpy.ndarray of shape (n, 1) of normalized closes
        stats: numpy.ndarray of shape (n, 2) of per-window mean and std

    Returns:
        numpy.ndarray of shape (n,) of prices in USD
    """
    return values[:, 0] * stats[:, 1] + stats[:, 0]


def evaluate(model, splits, name='test'):
    """Scores the model against a naive persistence baseline.

    The baseline predicts that the next hour closes exactly where the last
    one did. On a near random walk it is a genuinely hard baseline to beat,
    which is the point of reporting it.

    Args:
        model: the trained tf.keras.Model
        splits: dict returned by load_splits
        name: which split to score

    Returns:
        a dict of metrics in USD
    """
    x, y, stats, _ = splits[name]
    pred = model.predict(make_dataset(x, y), verbose=0)
    pred_usd = denormalize(pred, stats)
    true_usd = denormalize(y, stats)
    base_usd = denormalize(x[:, -1, :1], stats)
    err, base_err = pred_usd - true_usd, base_usd - true_usd
    prev = base_usd
    direction = np.mean(np.sign(pred_usd - prev) == np.sign(true_usd - prev))
    return {
        'mae_usd': float(np.mean(np.abs(err))),
        'rmse_usd': float(np.sqrt(np.mean(err ** 2))),
        'baseline_mae_usd': float(np.mean(np.abs(base_err))),
        'baseline_rmse_usd': float(np.sqrt(np.mean(base_err ** 2))),
        'directional_accuracy': float(direction),
        'n': int(len(y))
    }


def main(data='btc_hourly.npz', model_path='btc_model.keras',
         history_path='history.json', preds_path='predictions.npz'):
    """Trains the model, evaluates it and saves the artifacts."""
    splits = load_splits(data)
    x_train, y_train = splits['train'][0], splits['train'][1]
    x_val, y_val = splits['val'][0], splits['val'][1]
    train_ds = make_dataset(x_train, y_train, training=True)
    val_ds = make_dataset(x_val, y_val)

    model = build_model(x_train.shape[1], x_train.shape[2])
    model.summary()
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=[stop], verbose=2)

    metrics = evaluate(model, splits)
    print(json.dumps(metrics, indent=2))
    model.save(model_path)
    with open(history_path, 'w') as f:
        json.dump({'history': {k: [float(v) for v in vals]
                               for k, vals in history.history.items()},
                   'metrics': metrics}, f, indent=2)
    x_test, y_test, stats_test, t_test = splits['test']
    pred = model.predict(make_dataset(x_test, y_test), verbose=0)
    np.savez_compressed(preds_path, pred=denormalize(pred, stats_test),
                        true=denormalize(y_test, stats_test),
                        baseline=denormalize(x_test[:, -1, :1], stats_test),
                        t=t_test)
    return metrics


if __name__ == '__main__':
    main()
