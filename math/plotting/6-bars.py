#!/usr/bin/env python3
"""
Module that plots a stacked bar graph of fruit quantities per person.
Shows distribution of apples, bananas, oranges, and peaches for each person.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Define colors and fruit names
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
people = ['Farrah', 'Fred', 'Felicia']

# Create figure
plt.figure()

# Create x-axis positions for the bars
x = np.arange(len(people))

# Plot stacked bars
bottom = np.zeros(3)
for i in range(4):
    plt.bar(x, fruit[i], width=0.5, label=fruit_names[i],
            bottom=bottom, color=colors[i])
    bottom += fruit[i]

# Set labels, title, and ticks
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(x, people)
plt.yticks(range(0, 81, 10))
plt.ylim(0, 80)
plt.legend()

plt.show()
