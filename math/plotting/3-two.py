#!/usr/bin/env python3
"""
Module that plots exponential decay of two radioactive elements.
Displays C-14 as dashed red line and Ra-226 as solid green line.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.figure()
plt.plot(x, y1, '--r', label='C-14')
plt.plot(x, y2, '-g', label='Ra-226')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()
