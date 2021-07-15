from dask.distributed import Client

client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='2GB')

import dask_ml.datasets
import dask_ml.cluster
import matplotlib.pyplot as plt
print('after import')
X, y = dask_ml.datasets.make_blobs(n_samples=1000000,
                                   chunks=100000,
                                   random_state=0,
                                   centers=3)
#X = X.persist()
print('after persist', X)

km = dask_ml.cluster.KMeans(n_clusters=3, init_max_iter=2, oversampling_factor=10)
km.fit(X)
print("after fit")
fig, ax = plt.subplots()
ax.scatter(X[::1000, 0], X[::1000, 1], marker='.', c=km.labels_[::1000],
           cmap='viridis', alpha=0.25);

plt.show()