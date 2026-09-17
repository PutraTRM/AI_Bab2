# Label Spreading with 3 Groups

import numpy as np
from sklearn.semi_supervised import LabelSpreading

# Data points
X = np.array([
    # Group 1
    [1, 2, 3],
    [1, 4, 2],
    [1, 0, 3],

    # Group 2
    [10, 2, 4],
    [9, 4, 3],
    [11, 0, 2],

    # Group 3
    [5, 8, 7],
    [6, 9, 8],
    [4, 8, 9]
])

# Only one point in each group is labeled
# -1 means unlabeled
labels = np.array([
    0, -1, -1,    # Group 1
    1, -1, -1,    # Group 2
    2, -1, -1     # Group 3
])

# Label Spreading
label_spread = LabelSpreading(
    kernel='knn',
    alpha=0.8
)

label_spread.fit(X, labels)

# Get predicted labels
output_labels = label_spread.transduction_

print("Output labels:")
print(output_labels)