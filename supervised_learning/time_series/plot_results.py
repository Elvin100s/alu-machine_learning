#!/usr/bin/env python3
"""Draws the figures used to report the BTC forecasting results."""
import datetime
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


INK = '#1c1c1c'
ACCENT = '#c2410c'
MUTED = '#8a8a8a'


def style(ax, title, xlabel, ylabel):
    """Applies a consistent look to an axis."""
    ax.set_title(title, color=INK, fontsize=13, pad=12, loc='left')
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#d9d9d9')
    ax.grid(True, color='#ececec', linewidth=0.8)
    ax.set_axisbelow(True)


def plot_history(path, out):
    """Plots training and validation loss per epoch."""
    with open(path) as handle:
        hist = json.load(handle)['history']
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    epochs = np.arange(1, len(hist['loss']) + 1)
    ax.plot(epochs, hist['loss'], color=INK, linewidth=2, label='training')
    ax.plot(epochs, hist['val_loss'], color=ACCENT, linewidth=2,
            label='validation')
    best = int(np.argmin(hist['val_loss']))
    ax.scatter([best + 1], [hist['val_loss'][best]], color=ACCENT, zorder=5)
    ax.annotate('best epoch {}'.format(best + 1),
                (best + 1, hist['val_loss'][best]),
                textcoords='offset points', xytext=(8, 10),
                color=ACCENT, fontsize=9)
    style(ax, 'Training and validation loss (MSE, normalized units)',
          'epoch', 'loss')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close(fig)


def plot_predictions(path, out, hours=336):
    """Plots predicted against actual close prices over the test window."""
    data = np.load(path)
    pred, true = data['pred'][-hours:], data['true'][-hours:]
    stamps = [datetime.datetime.utcfromtimestamp(t)
              for t in data['t'][-hours:]]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    ax.plot(stamps, true, color=INK, linewidth=2, label='actual')
    ax.plot(stamps, pred, color=ACCENT, linewidth=1.6, label='predicted')
    style(ax, 'Predicted vs actual BTC-USD close, final {} test hours'.format(
        hours), 'date (UTC)', 'USD')
    ax.legend(frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close(fig)


def plot_errors(path, out):
    """Plots the error distribution against the persistence baseline."""
    data = np.load(path)
    err = data['pred'] - data['true']
    base = data['baseline'] - data['true']
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    bins = np.linspace(min(err.min(), base.min()),
                       max(err.max(), base.max()), 80)
    ax.hist(base, bins=bins, color=MUTED, alpha=0.65,
            label='persistence baseline')
    ax.hist(err, bins=bins, color=ACCENT, alpha=0.75, label='LSTM')
    style(ax, 'Hourly forecast error distribution', 'error (USD)', 'hours')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close(fig)


def plot_zoom(path, out, hours=48):
    """Zooms in far enough to show the model tracking one hour behind.

    Over hundreds of hours the prediction looks almost perfect. That is
    the illusion: at close range it is visibly a copy of the previous
    hour, shifted right. Plotting the persistence baseline alongside it
    makes the overlap plain.
    """
    data = np.load(path)
    pred, true = data['pred'][-hours:], data['true'][-hours:]
    base = data['baseline'][-hours:]
    stamps = [datetime.datetime.utcfromtimestamp(t)
              for t in data['t'][-hours:]]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    ax.plot(stamps, true, color=INK, linewidth=2.4, label='actual')
    ax.plot(stamps, pred, color=ACCENT, linewidth=1.8, label='LSTM')
    ax.plot(stamps, base, color=MUTED, linewidth=1.4, linestyle='--',
            label='persistence (last close)')
    style(ax, 'The same forecast at {}-hour zoom'.format(hours),
          'date (UTC)', 'USD')
    ax.legend(frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    plot_history('history.json', 'fig_training_loss.png')
    plot_predictions('predictions.npz', 'fig_predictions.png')
    plot_errors('predictions.npz', 'fig_errors.png')
    plot_zoom('predictions.npz', 'fig_zoom.png')
    print('wrote fig_training_loss.png, fig_predictions.png, '
          'fig_errors.png, fig_zoom.png')
