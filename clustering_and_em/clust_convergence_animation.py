import numpy as np
import matplotlib.pyplot as plt
import os
from helper_functions import *

def free_energy(x, K, qs, mus, pis, covs):
    
    x_ = np.stack([x]*K, axis = 1)
    dot_mat = np.swapaxes((x_ - mus).dot(np.linalg.inv(covs).T)[:, 0, :, :], 1, 2)
    exponent = -0.5*np.sum(dot_mat*(x_ - mus), axis = -1)
    log_joint = exponent + np.log(pis) - 0.5*np.log(2*np.pi*np.linalg.det(covs))
    
    f_energy = np.sum(qs*log_joint - qs*np.log(qs))
    
    return f_energy
    

def mog_EM(x, K, q_init, mu_init, pi_init, cov_init, max_steps):
    
    q, mu, pi, cov = q_init.copy(), mu_init.copy(), pi_init.copy(), cov_init.copy()
    f_energy = free_energy(x, K, q_init, mu_init, pi_init, cov_init)
    qs, mus, pis, covs, f_energies = [q.T], [mu], [pi], [cov], [f_energy]

    for n in range(max_steps + 1):
        
        # E-step
        prec = np.linalg.inv(cov)
        exponent = np.einsum('ijk, nik -> nij', prec, xs - mu)
        exponent = np.einsum('nij, nij -> ni', xs - mu, exponent)
        exp_term = np.exp(-1/2*exponent)
        q = (exp_term*pi/(2*np.pi*np.linalg.det(cov))**0.5).T
        q = q/q.sum(axis = 0)
        
        qs.append(q), mus.append(mu), pis.append(pi), covs.append(cov)
        f_energies.append(free_energy(x, K, q.T, mu, pi, cov))

        # M-step
        N_k = np.sum(q, axis = 1)
        mu = (q.dot(x).T/N_k).T

        pi = N_k/x.shape[0]

        cov = np.einsum('ijk, ijl -> ijkl', xs - mu, xs - mu)
        cov = np.einsum('ij, jikl -> ijkl', q, cov)
        cov = (np.sum(cov, axis = 1).T/N_k).T
        
        qs.append(q), mus.append(mu), pis.append(pi), covs.append(cov)
        f_energies.append(free_energy(x, K, q.T, mu, pi, cov))
    
    return qs, mus, pis, covs, f_energies

x = np.load('clustering_2d.npy') # load the 2d clustering dataset
K = 3 
q_init = np.ones(shape = (x.shape[0], K))/K # initial responsibilities = 1/K
mu_init = [np.array([-0.5, -1.5]), np.array([-0.5, 1]), np.array([1, 0.75])] # initial means judged by eye
cov_init = np.stack([np.eye(2)]*K)  # initial covariances equal to identity
pi_init = np.ones(shape = (K,))/K # initial cluster probabilities = 1/K
xs = np.stack([x]*K, axis = 1) # stack K copies of the data to do arithmetic conveniently
max_steps = 8 # maximum number of EM steps

qs, mus, pis, covs, f_energies = mog_EM(x, K, q_init, mu_init, pi_init, cov_init, max_steps) # apply EM

x_, y_ = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 250)) # x, y where to evaluate gaussians
grid = np.stack([x_, y_], axis = -1) # x, y grid to use for calculation of gaussians
colors = ['red', 'green', 'blue']

plt.figure(figsize = (10, 5))
for i in range(2*max_steps):
    
    plt.clf()
    
    plt.subplot(1, 2, 1)
    q, mu, pi, cov = qs[i], mus[i], pis[i], covs[i] # parameters of the i^th EM step
    
    for k in range(K):
        mu_, cov_ = mu[k], cov[k] # mean/covariance of the k^th gaussian cluster

        exp_ = np.exp(-0.5*np.sum((grid - mu_).dot(np.linalg.inv(cov_))*(grid - mu_),
                                  axis = -1)) # evaluate the k^th gaussian on the grid
        plt.contour(x_, y_, exp_, 4, colors = colors[k]) # plot the gaussian

    plt.scatter(x[:, 0], x[:, 1], s = 20, c = q.T, edgecolor = 'black') # show data, coloured by membership q
    beautify_plot({"title":"Iter. {} ({} step)".format((i+1)//2, 'EM'[(i + 1)%2]), "x":"$x$", "y":"$y$"})

    plt.subplot(1, 2, 2)
    plt.plot(f_energies, color = 'black')

    
    plt.tight_layout()
    plt.show()
