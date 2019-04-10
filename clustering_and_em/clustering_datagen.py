import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

cov_0 = np.array([[1.5, 0],
                  [0, 1.5]])

cov_1 = np.array([[1, 0.9],
                  [0.9, 1]])

cov_2 = np.array([[1.5, 0.5],
                  [0.5, 1.5]])

covs = np.stack([cov_0, cov_1, cov_2], axis = 0)

mu_0 = np.array([[10, 10]]).T

mu_1 = np.array([[5, 13]]).T

mu_2 = np.array([[6, 2]]).T

mus = np.stack([mu_0, mu_1, mu_2], axis = 0)

xs = []
rs = []

for n in range(100):

    r = np.random.choice([0, 1, 2], p = [0.4, 0.3, 0.3])
    rs.append(r)

    cov = covs[r]
    mu = np.reshape(mus[r], (-1))

    xs.append(np.random.multivariate_normal(mu, cov))
    
xs = np.array(xs)
xs[:, 0] = (xs[:, 0] - xs[:, 0].mean())/xs[:, 0].var()**0.5
xs[:, 1] = (xs[:, 1] - xs[:, 1].mean())/xs[:, 1].var()**0.5

#np.save('clustering_2d.npy', xs)
#np.save('clustering_2d_classes.npy', np.array(rs))
