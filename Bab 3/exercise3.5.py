# Exercise 3.7 Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import pickle

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Create Naive Bayes model
clf = GaussianNB()

# Train the model
clf.fit(X, y)

# Save the trained model to a file
with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

print("Model saved successfully.")

# Load the model from the file
with open('naive_bayes_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully.")

# Make a prediction using the loaded model
p = loaded_model.predict([[5.4, 3.2, 1.5, 0.2]])

print("Prediction:", p)