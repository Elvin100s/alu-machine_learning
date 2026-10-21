# Hidden Markov Models

Implementation of Markov chains and hidden Markov models (HMMs) in Python
with NumPy only.

## Files

| File | Description |
| --- | --- |
| [0-markov_chain.py](0-markov_chain.py) | State probabilities of a Markov chain after `t` iterations |
| [1-regular.py](1-regular.py) | Steady state probabilities of a regular Markov chain |
| [2-absorbing.py](2-absorbing.py) | Whether a Markov chain is absorbing |
| [3-forward.py](3-forward.py) | Forward algorithm: likelihood and forward path probabilities |
| [4-viterbi.py](4-viterbi.py) | Viterbi algorithm: most likely sequence of hidden states |
| [5-backward.py](5-backward.py) | Backward algorithm: likelihood and backward path probabilities |
| [6-baum_welch.py](6-baum_welch.py) | Baum-Welch: unsupervised training of transition and emission matrices |

## Requirements

* Python 3.5+
* NumPy 1.15+

## Usage

Each module exposes a single function and can be imported directly:

```python
#!/usr/bin/env python3
import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

P = np.array([[0.25, 0.2, 0.25, 0.3],
              [0.2, 0.3, 0.2, 0.3],
              [0.25, 0.25, 0.4, 0.1],
              [0.3, 0.3, 0.1, 0.3]])
s = np.array([[1, 0, 0, 0]])

print(markov_chain(P, s, 300))
```

The `*-main.py` files are the example scripts of the project and can be run
with `./0-main.py`, `./1-main.py`, and so on.
