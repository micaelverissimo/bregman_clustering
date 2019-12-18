import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class base_kmeans(object):
    def __init__(self, n_clusters, seed=13):
        self.seed       = seed
        self.n_clusters = n_clusters

        np.random.seed(self.seed)

    def get_n_clusters(self):
        return self.n_clusters

    def get_centroids(self):
        return self.centroids

    def get_first_centroids(self):
        return self.first_centroids

    def get_n_dim(self):
        return self.n_dim

    def fit(self, X_data, breg_div='euclidean', n_iter=10, tol=1e-3):
        # begin: initialize the centroids
        self.X          = X_data
        self.breg_div   = breg_div
        self.n_iter     = n_iter
        self.n_dim      = X_data.shape[1]
        self.centroids  = np.random.uniform(low=np.min(self.X, axis=0), high=np.max(self.X,axis=0),
                                            size=(self.n_clusters, self.n_dim))
        #self.F_in       = np.zeros(self.n_clusters)
        self.labels     = None
        print('Begin K-means... ')
        self.first_centroids = self.centroids
        for i_iter in range(n_iter):
            new_centroids = np.zeros_like(self.centroids)
            print('Iteraction: %i' %(i_iter+1))
            print('Calculating the %s divergence between the data and the centroids...' %(self.breg_div))
            if self.breg_div == 'itakura-saito':
                dist = distance.cdist(self.X, self.centroids,
                                      metric=lambda u, v: ((u/v)-np.log(u/v)-1).sum())
            elif self.breg_div == 'exp':
                dist = distance.cdist(self.X, self.centroids,
                                      metric=lambda u, v: (np.exp(u)-np.exp(v)-(u-v)*np.exp(v)).sum())
            elif self.breg_div == 'gen_kl':
                dist = distance.cdist(self.X, self.centroids,
                                      metric=lambda u, v: ((u*np.log(u/v)).sum()-(u-v).sum()).sum())
            else:
                dist = distance.cdist(self.X, self.centroids, metric=self.breg_div)

            # Classification Step
            self.labels = np.argmin(dist, axis=1)
            # Renewal Step
            for icluster in range(self.centroids.shape[0]):
                if self.X[self.labels==icluster].shape[0] != 0:
                    new_centroids[icluster] = np.mean(self.X[self.labels==icluster], axis=0)
                else:
                    new_centroids[icluster] = self.centroids[icluster]
            self.centroids = new_centroids
