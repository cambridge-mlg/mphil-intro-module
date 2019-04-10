import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

cov_0 = np.array([[0.2, 0],
                  [0, 0.2]])

cov_1 = np.array([[10, 0],
                  [0, 0.1]])

cov_2 = np.array([[0.2, 0],
                  [0, 0.2]])

covs = np.stack([cov_0, cov_1, cov_2], axis = 0)

mu_0 = np.array([[-4, -2.5]]).T

mu_1 = np.array([[0, 0]]).T

mu_2 = np.array([[4, 2.5]]).T

mus = np.stack([mu_0, mu_1, mu_2], axis = 0)

xs = []

for n in range(100):

    r = np.random.choice([0, 1, 2], p = [0.25, 0.5, 0.25])

    cov = covs[r]
    mu = np.reshape(mus[r], (-1))

    xs.append(np.random.multivariate_normal(mu, cov))
    
xs = np.array(xs)

np.save('clust_kmeans_pathology.npy', xs)
