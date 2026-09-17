# Example 3.20 K-Means Clustering
# Modified by adding two points to each group

from sklearn.cluster import KMeans
import numpy as np

# Original data + two additional points for each group
X = np.array([
    # Group 1
    [1, 2, 3],
    [1, 4, 2],
    [1, 0, 3],
    [2, 2, 3],
    [2, 3, 2],

    # Group 2
    [10, 2, 4],
    [9, 4, 3],
    [11, 0, 2],
    [10, 3, 3],
    [10, 1, 3]
])

# K-Means clustering
kmeans = KMeans(
    n_clusters=2,
    random_state=0
).fit(X)

# Print cluster labels
print("Cluster labels:")
print(kmeans.labels_)

# Print cluster centers
print("Cluster centers:")
print(kmeans.cluster_centers_)

# Predict a new data point
print("Prediction:")
print(kmeans.predict([[12, 3, 1]]))