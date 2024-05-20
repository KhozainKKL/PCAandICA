import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from mpl_toolkits.mplot3d import Axes3D
# import relevant libraries for 3d graph

data = load_breast_cancer()
data.keys()

print(data['target_names'])
print(data['feature_names'])

# construct a dataframe using pandas
df1 = pd.DataFrame(data['data'], columns=data['feature_names'])

# Scale data before applying PCA
scaling = StandardScaler()

# Use fit and transform method
scaling.fit(df1)
Scaled_data = scaling.transform(df1)

# Set the n_components=3
principal = PCA(n_components=3)
principal.fit(Scaled_data)
x = principal.transform(Scaled_data)

# Check the dimensions of data after PCA
print(x.shape)

plt.figure(figsize=(10, 10))
plt.scatter(x[:, 0], x[:, 1], c=data['target'], cmap='plasma')

fig = plt.figure(figsize=(10, 10))

# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')

print(principal.explained_variance_ratio_)
axis.scatter(x[:, 0], x[:, 1], x[:, 2], c=data['target'], cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)
plt.show()



