# Exercise 3.15

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 8, 10, 12, 15, 16, 19, 21]

plt.plot(x, y, marker='o', label='Data')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('X and Y Data')
plt.legend()
plt.grid(True)
plt.show()