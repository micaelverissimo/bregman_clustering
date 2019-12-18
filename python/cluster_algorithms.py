import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

from sklearn import datasets
from scipy.spatial import Voronoi, voronoi_plot_2d
import time

class base_kmeans(object):
    def __init__(self, n_clusters, seed=None):
        self.seed            = seed
        self.n_clusters      = n_clusters
        # set the selected seed
        np.random.seed(self.seed)
        self.dict_breg_divs  = {
            'itakura-saito' : lambda u, v: ((u/v)-np.log(u/v)-1).sum(),
            'exp'           : lambda u, v: (np.exp(u)-np.exp(v)-(u-v)*np.exp(v)).sum(),
            'gen_kl'        : lambda u, v: ((u*np.log(u/v)).sum()-(u-v).sum()).sum(),
            'gen_kls'       : lambda u, v: 0.5*(((u*np.log(u/v)).sum()-(u-v).sum()).sum() + ((v*np.log(v/u)).sum()-(v-u).sum()).sum()),
            'gen_js'        : lambda u, v: 0.5*(((u*np.log(u/(.5*(u+v)))).sum()-(u-.5*(u+v)).sum()).sum() + \
                ((v*np.log(v/(.5*(u+v)))).sum() - (v-(.5*(u+v))).sum()).sum()),
            'euclidean'     : 'euclidean'
        } 
        
    def get_n_clusters(self):
        return self.n_clusters
    
    def get_centroids(self):
        return self.centroids
    
    def get_first_centroids(self):
        return self.first_centroids
    
    def get_n_dim(self):
        return self.n_dim
    
    def get_sum_total_div(self):
        return np.array(self.sum_total_div)
    
    def get_last_iter(self):
        return self.last_iter
    
    def get_labels(self):
        return self.labels

    def classification_and_renewal(self, distances):
        cluster_div   = []
        new_centroids = np.zeros_like(self.centroids)
        # Classification Step
        self.labels = np.argmin(distances, axis=1)
        # Renewal Step
        for icluster in range(self.centroids.shape[0]):
            if self.X[self.labels==icluster].shape[0] != 0:
                new_centroids[icluster] = np.mean(self.X[self.labels==icluster], axis=0)
                # Calculate the div inter cluster
                cluster_div.append(distance.cdist(self.X[self.labels==icluster], self.centroids[icluster][np.newaxis],
                                                  metric=self.dict_breg_divs[self.breg_div]).sum())
            else:
                new_centroids[icluster] = self.centroids[icluster]
        return np.array(cluster_div).sum(), new_centroids

    def predict_cluster(self, X):
        dist = distance.cdist(X, self.centroids,
                              metric=self.dict_breg_divs[self.breg_div])
        predicted_label = np.argmin(dist, axis=1)
        return predicted_label
    
    def fit(self, X_data, breg_div='euclidean', n_iter=10, tol=1e-3, debug=False):
        np.random.seed(self.seed)
        # begin: initialize the centroids
        self.tol           = tol
        self.X             = X_data
        self.breg_div      = breg_div
        self.n_iter        = n_iter
        self.n_dim         = X_data.shape[1]
        self.centroids     = np.random.uniform(low=np.min(self.X, axis=0), high=np.max(self.X,axis=0),
                                            size=(self.n_clusters, self.n_dim))
        self.sum_total_div = []
        self.labels        = None
        if debug:
            print('Begin K-means using %s divergence... ' %(self.breg_div))
        self.first_centroids = self.centroids
        for i_iter in range(n_iter):
            if debug:
                print('Iteraction: %i' %(i_iter+1))
            dist = distance.cdist(self.X, self.centroids,
                                  metric=self.dict_breg_divs[self.breg_div])
            # Classification and Renewal step
            clust_div, new_centers = self.classification_and_renewal(dist)
            # Check convergence
            self.sum_total_div.append(clust_div)
            if i_iter == 0:
                stop_criteria = self.sum_total_div[-1]
            else:
                stop_criteria = np.abs(self.sum_total_div[-1] - self.sum_total_div[-2])
            if stop_criteria < self.tol:
                # Jut to log the number of iteractions
                self.last_iter = i_iter+1
                print('The conversion criteria was reached... Stopping!')
                break
            else:
                self.centroids = new_centers
                self.last_iter = i_iter+1
            zeros = self.centroids == 0.
            self.centroids[zeros] = 1e-1