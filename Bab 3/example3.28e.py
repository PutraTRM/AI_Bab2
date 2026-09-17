import pandas as pd
import matplotlib.pyplot as plt

from pycaret import regression
from sklearn import datasets

# Load dataset
diabetes = datasets.load_diabetes(as_frame=True)

data = diabetes.data
data['target'] = diabetes.target

# Setup PyCaret
regression.setup(
    data=data,
    target='target'
)

# Membandingkan model
models = regression.compare_models(
    n_select=10,
    sort='R2'
)

# Tampilkan hasil model
print(models)

# Plot R-Squared
plt.figure(figsize=(10, 5))

plt.plot(
    models.index,
    models['R2'],
    '-s'
)

plt.xlabel('Model')
plt.ylabel('R-Squared')
plt.title('Perbandingan R-Squared Setiap Model')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()