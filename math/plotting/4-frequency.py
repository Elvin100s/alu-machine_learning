#!/usr/bin/env python3
"""
Module that plots a histogram of student project grades.
Shows frequency distribution with bins every 10 units.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.figure()
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
