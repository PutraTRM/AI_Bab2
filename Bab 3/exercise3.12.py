# Exercise - Multiple Linear Regression using Linnerud Dataset

from sklearn import linear_model
from sklearn.datasets import load_linnerud
import numpy as np

# Load Linnerud dataset
data = load_linnerud()

x = data.data
y = data.target

# Create Linear Regression model
reg = linear_model.LinearRegression()

# Train the model
reg.fit(x, y)

# Print coefficients
print('Coefficients:')
print(reg.coef_)

# Print intercept
print('Intercept:')
print(reg.intercept_)

# Prediction
pred = reg.predict([[5, 30, 60]])

print('Prediction:')
print(pred)