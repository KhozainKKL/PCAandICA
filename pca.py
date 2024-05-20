import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D

# Загрузка данных Iris
data = load_iris()
X = data.data
y = data.target
#
# # Применение PCA для уменьшения размерности до 2 компонент
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
#
# # Создание графиков
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#
# # Исходные данные (2D-график с первыми двумя признаками)
# scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
# ax1.set_title('Исходные данные (2D)')
# ax1.set_xlabel(data.feature_names[0])
# ax1.set_ylabel(data.feature_names[1])
# legend1 = ax1.legend(*scatter1.legend_elements(), title="Классы")
# ax1.add_artist(legend1)
#
# # Данные после применения PCA (2D-график)
# scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
# ax2.set_title('Данные после применения PCA (2D)')
# ax2.set_xlabel('Principal Component 1')
# ax2.set_ylabel('Principal Component 2')
# legend2 = ax2.legend(*scatter2.legend_elements(), title="Классы")
# ax2.add_artist(legend2)
#
# plt.show()

""" из 3D в 2D"""

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Применение PCA для уменьшения размерности до 3 компонент
pca = PCA(n_components=3)
pca.fit(X)

print('Variance ratio: ' + str(pca.explained_variance_ratio_))
print('Singular values: ' + str(pca.singular_values_))

# Преобразование данных с использованием PCA
pca_X = pca.transform(X)
print('Values after PCA: ' + str(pca_X))

# Создание цветовой карты
colors = ['red', 'green', 'blue']
color_map = np.array([colors[label] for label in y])

fig = plt.figure(figsize=(14, 6))

# Исходные данные (3D-график)
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color_map, edgecolor='k')
ax1.set_title('Исходные данные (3D)')
ax1.set_xlabel(data.feature_names[0])
ax1.set_ylabel(data.feature_names[1])
ax1.set_zlabel(data.feature_names[2])

# Данные после применения PCA (2D-график)
ax2 = fig.add_subplot(122)
ax2.scatter(pca_X[:, 0], pca_X[:, 1], c=color_map, edgecolor='k')
ax2.set_title('Данные после применения PCA (2D)')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

plt.show()