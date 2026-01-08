#!/usr/bin/env python3
"""
Module that plots exponential decay of C-14 as a line graph.
Y-axis is logarithmically scaled to show the decay pattern.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.figure()
plt.plot(x, y)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.yscale('log')
plt.xlim(0, 28650)
plt.show()
