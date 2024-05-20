import numpy as np
from sklearn.decomposition import FastICA

# array = np.random.normal(size=(1000,5))
np.random.seed(0)
X = np.random.uniform(low=0, high=10, size=(5, 10))

print('Before ICA: ' + str(X))
print('Shape' + str(X.shape))

ica = FastICA(n_components=3)
S = ica.fit_transform(X)

print('After ICA: ' + str(S))
print('Shape' + str(S.shape))

