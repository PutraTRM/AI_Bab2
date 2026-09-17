# Exercise 3.11 Decision Tree Classification - Wine Dataset

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load Wine dataset
X, y = load_wine(return_X_y=True)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=0
)

# Create Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Calculate total and correctly labeled points
N = y_test.shape[0]
C = (y_test == y_pred).sum()

print("Total points: %d" % N)
print("Correctly labeled points: %d" % C)

# Calculate accuracy
accuracy = C / N * 100
print("Accuracy: %.2f%%" % accuracy)