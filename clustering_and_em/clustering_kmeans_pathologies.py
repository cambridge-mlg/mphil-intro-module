import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
set_notebook_preferences()

np.random.seed(0)
colors  = np.array(['r', 'g', 'b'])


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
rs = []

for n in range(100):

    r = np.random.choice([0, 1, 2], p = [0.25, 0.5, 0.25])
    rs.append(r)
    
    cov = covs[r]
    mu = np.reshape(mus[r], (-1))

    xs.append(np.random.multivariate_normal(mu, cov))
    
x = np.array(xs)
r = np.array(rs)

def k_means(x, K, max_steps, mu_init):
    
    N, D = x.shape # N: number of datapoints, D: number of input dimensions
    mu = mu_init.copy() # copy cluster centers to avoid mutation

    s = np.zeros(shape = (N, K)) # set all membership indices to 0
    assignments = np.random.choice(np.arange(0, K), N)
    s[np.arange(s.shape[0]), assignments] = 1
    
    x_stacked = np.stack([x]*K, axis = 1)
    losses = [np.sum(s*np.sum((x_stacked - mu)**2, axis = 2))]
    converged = False
    
    for i in range(max_steps):

        mus = (s.T).dot(x)
        s_sum = s.sum(axis = 0).reshape((-1, 1))
        s_sum[np.where(s_sum < 1)] = 1
        mus /= s_sum

        distances = np.sum((x_stacked - mus)**2, axis = 2)
        min_idx = np.argmin(distances, axis = 1)
        s_prev = s.copy()
        s = np.zeros_like(s)
        s[np.arange(s.shape[0]), min_idx] = 1

        losses.append(np.sum(s*np.sum((x_stacked - mus)**2, axis = 2)))
        
        if np.prod(np.argmax(s, axis = 1) == np.argmax(s_prev, axis = 1)):
            break
        
    return s, mus, losses


s, mus, cost = k_means(x, 3, 20, np.random.rand(3, 2)*4 - 2)

plt.figure(figsize = (12, 4))
plt.subplot(131)
plt.scatter(x[:, 0], x[:, 1], marker = 'o',
            color = 'white', edgecolor = 'black', zorder = 1)
beautify_plot({"title":"Pathological dataset", "x":"$x_1$", "y":"$x_2$"})
plt.xticks(np.arange(-6, 7, 3)), plt.yticks(np.arange(-4, 5, 2))

plt.subplot(132)
plt.scatter(x[:, 0], x[:, 1], marker = 'o',
            color = colors[r], edgecolor = 'black', zorder = 1)
beautify_plot({"title":"Good clustering", "x":"$x_1$", "y":"$x_2$"})
plt.xticks(np.arange(-6, 7, 3)), plt.yticks(np.arange(-4, 5, 2))


plt.subplot(133)
for idx, mu in enumerate(mus):
    plt.scatter(mu[0], mu[1], marker = '^', color = colors[idx], s = 200,
                edgecolor = 'black', zorder = 2, linewidth = '2')
    
    points_in_class = x[np.where(s[:, idx] == 1)[0], :]
    
    plt.scatter(points_in_class[:, 0], points_in_class[:, 1], marker = 'o',
                color = colors[idx], edgecolor = 'black', zorder = 1)
    
beautify_plot({"title":"k-means result", "x":"$x_1$", "y":"$x_2$"})
plt.xticks(np.arange(-6, 7, 3)), plt.yticks(np.arange(-4, 5, 2))
plt.tight_layout()
plt.savefig('clust_kmeans_pathology.svg')
plt.show()
