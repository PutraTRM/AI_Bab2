# Example 3.20 K-Means Clustering
# Using sklearn.datasets.make_blobs()

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(
    n_samples=300,
    centers=2,
    n_features=3,
    random_state=0
)

# Perform K-Means clustering
kmeans = KMeans(
    n_clusters=2,
    random_state=0
).fit(X)

# Print cluster labels
print(kmeans.labels_)

# Print cluster centers
print(kmeans.cluster_centers_)

# Predict cluster for a new data point
print(kmeans.predict([[12, 3, 1]]))