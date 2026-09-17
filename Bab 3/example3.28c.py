# EXAMPLE 3.28C THE LAZYPREDICT.IPYNB PROGRAM (PART 3)

from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Dataset
data = load_iris()

X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# LazyPredict
clf = LazyClassifier()

models, predictions = clf.fit(
    X_train,
    X_test,
    y_train,
    y_test
)

# Tampilkan hasil
print(models)

# Grafik Accuracy
plt.figure(figsize=(10, 5))

plt.plot(
    models.index,
    models['Accuracy'],
    '-s'
)

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Perbandingan Accuracy Setiap Model')

plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()

plt.show()