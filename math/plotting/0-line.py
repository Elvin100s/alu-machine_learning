#!/usr/bin/env python3
"""
Module that plots y as a cubic line graph.
y is plotted as a solid red line with x-axis ranging from 0 to 10.
"""
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.figure()
plt.plot(y, color='red')
plt.xlim(0, 10)
plt.show()
