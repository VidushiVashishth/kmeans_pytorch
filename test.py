import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import KMeans

# set random seed
np.random.seed(123)

# dimensions, num clusters
dims, num_clusters = 2, 3

# data sizes
data_sizes = [1000]

gpu_times = []
cpu_times = []
y = np.random.randn(1, dims).astype("float")
y = torch.tensor(y)

for data_size in data_sizes:
    print(f'\ndata size: {data_size}')

    # data
    x = np.random.randn(data_size, dims).astype('f') / 6
    x = torch.from_numpy(x)

    # gpu
    cluster_ids, cluster_centers = kmeans(X=x, num_clusters=num_clusters, device=torch.device('cuda:0'))
    print(cluster_ids)
    print(type(cluster_ids))
    print(cluster_ids.shape)
    print(cluster_centers)
    print(type(cluster_centers))
    print(cluster_centers.shape)
    result = kmeans_predict(y, cluster_centers)
    print(result)
    print(type(result))
    print(result.shape)

    # cpu
    km = KMeans(n_clusters=num_clusters).fit(x)
    print(km.cluster_centers_)
    print(type(km.cluster_centers_))
    print(km.cluster_centers_.shape)
    result = km.predict(y)
    print(result)
    print(type(result))
    print(result.shape)



# plot
# plt.figure(figsize=(6, 3), dpi=160)
# plt.plot(data_sizes, gpu_times, marker='o', label='gpu', color='xkcd:vermillion')
# plt.plot(data_sizes, cpu_times, marker='o', label='cpu', color='xkcd:neon blue')
# plt.xticks(data_sizes)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.2)
# plt.xlabel('data size', fontsize=14)
# plt.ylabel('time (s)', fontsize=14)
# plt.savefig('test.png')
