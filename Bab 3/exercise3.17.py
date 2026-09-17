# Example 3.24 - Ensemble1.py

from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Load Iris dataset
iris = datasets.load_iris()

X, y = iris.data[:, 1:3], iris.target


# Create classifiers
clf1 = LogisticRegression(random_state=1)

clf2 = RandomForestClassifier(
    n_estimators=50,
    random_state=1
)

clf3 = GaussianNB()

clf4 = SVC()

clf5 = KNeighborsClassifier(
    n_neighbors=5
)


# Create ensemble
eclf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('gnb', clf3),
        ('svc', clf4),
        ('knn', clf5)
    ],
    voting='hard'
)


# Test classifiers
for clf, label in zip(
    [clf1, clf2, clf3, clf4, clf5, eclf],
    [
        'Logistic Regression',
        'Random Forest',
        'Naive Bayes',
        'SVM',
        'K-Nearest Neighbor',
        'Ensemble'
    ]
):

    scores = cross_val_score(
        clf,
        X,
        y,
        scoring='accuracy',
        cv=5
    )

    print(
        "Accuracy: %0.2f (+/- %0.2f) [%s]"
        % (scores.mean(), scores.std(), label)
    )