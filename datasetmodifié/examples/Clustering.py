# %%
"""
In this example we will see how to use MiniSom to cluster the iris dataset.

First, let's load the data and train our SOM:
"""

# %%
from minisom import MiniSom
import numpy as np
import pandas as pd
import matplotlib as plt

data = pd.read_csv('NSL-KDD-Dataset\KDDTrain+.txt', 
                    names=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target'], usecols=[0, 5], 
                   sep='\t+', engine='python')
# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# Initialization and training
som_shape = (1, 3)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)

som.train_batch(data, 500, verbose=True)

# %%
"""
Now we will consider all the sample mapped into a specific neuron as a cluster. To identify each cluster more easily we will translate the bidimensional indexes of the neurons on the SOM into a monodimentional indexes:
"""

# %%
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

# %%
"""
We can plot each cluster with a different color:
"""

# %%
import matplotlib.pyplot as plt


# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=80, linewidths=35, color='k', label='centroid')
