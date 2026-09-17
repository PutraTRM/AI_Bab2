import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Names of classifiers
names = [
    "SVM",
    "Naive Bayes",
    "LDA",
    "QDA",
    "Decision Tree",
    "Random Forest",
    "Nearest Neighbors",
    "Neural Networks"
]

# Classifiers
classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000)
]

# Load Diabetes dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# Convert continuous target into 2 classes
# 0 = below median
# 1 = equal to or above median
y = (y >= np.median(y)).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=0
)

# Train and evaluate classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name + ": " + str(score))