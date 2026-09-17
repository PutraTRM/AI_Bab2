# Exercise 3.12 Random Forest Classification - Diabetes Data

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load Diabetes dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# Convert continuous target into 2 classes
# 0 = below median, 1 = above median
y = (y >= np.median(y)).astype(int)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=0
)

# Create Random Forest classifier
clf = RandomForestClassifier(
    random_state=0
)

# Train the model
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Print results
N = y_test.shape[0]
C = (y_test == y_pred).sum()

print("Total points: %d Correctly labeled points: %d" % (N, C))

# Accuracy
accuracy = C / N * 100
print("Accuracy: %.2f%%" % accuracy)