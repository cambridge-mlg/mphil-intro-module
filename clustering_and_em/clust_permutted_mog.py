import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
set_notebook_preferences()

def mog_EM(x, K, q_init, mu_init, pi_init, cov_init, max_steps):
    
    q, mu, pi, cov = q_init.copy(), mu_init.copy(), pi_init.copy(), cov_init.copy()
    qs, mus, pis, covs = [q.T], [mu], [pi], [cov]

    for n in range(max_steps + 1):
        
        # E-step
        prec = np.linalg.inv(cov)
        exponent = np.einsum('ijk, nik -> nij', prec, xs - mu)
        exponent = np.einsum('nij, nij -> ni', xs - mu, exponent)
        exp_term = np.exp(-1/2*exponent)
        q = (exp_term*pi/(2*np.pi*np.linalg.det(cov))**0.5).T
        q = q/q.sum(axis = 0)
        
        qs.append(q), mus.append(mu), pis.append(pi), covs.append(cov)

        # M-step
        N_k = np.sum(q, axis = 1)
        mu = (q.dot(x).T/N_k).T

        pi = N_k/x.shape[0]

        cov = np.einsum('ijk, ijl -> ijkl', xs - mu, xs - mu)
        cov = np.einsum('ij, jikl -> ijkl', q, cov)
        cov = (np.sum(cov, axis = 1).T/N_k).T
        
        qs.append(q), mus.append(mu), pis.append(pi), covs.append(cov)
    
    return qs, mus, pis, covs

x = np.load('clustering_2d.npy')
K = 3 
q_init = np.ones(shape = (x.shape[0], K))/K # initial responsibilities = 1/K
mu_init = [np.array([-0.5, 1]), np.array([1, 0.75]), np.array([-0.5, -1.5])] # initial means judged by eye
cov_init = np.stack([np.eye(2)]*K) # initial covariances equal to identity
pi_init = np.ones(shape = (K,))/K # initial cluster probabilities = 1/K
xs = np.stack([x]*K, axis = 1) # stack K copies of the data to do arithmetic conveniently
max_steps = 20 # maximum number of EM steps

qs, mus, pis, covs = mog_EM(x, K, q_init, mu_init, pi_init, cov_init, max_steps) # apply EM


x_, y_ = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 250)) # x, y where to evaluate gaussians
grid = np.stack([x_, y_], axis = -1) # x, y grid to use for calculation of gaussians
colors = ['red', 'green', 'blue']
axes = []

fig = plt.figure(figsize = (10, 5))
for idx, i in enumerate([0, 40]):
    
    plt.subplot(1, 2, idx + 1)
    q, mu, pi, cov = qs[i], mus[i], pis[i], covs[i] # parameters of the i^th EM step
    
    for k in range(K):
        mu_, cov_ = mu[k], cov[k] # mean/covariance of the k^th gaussian cluster

        exp_ = np.exp(-0.5*np.sum((grid - mu_).dot(np.linalg.inv(cov_))*(grid - mu_),
                                  axis = -1)) # evaluate the k^th gaussian on the grid
        plt.contour(x_, y_, exp_, 4, colors = colors[k]) # plot the gaussian

    plt.scatter(x[:, 0], x[:, 1], s = 50, c = q.T, edgecolor = 'black') # show data, coloured by membership q
    beautify_plot({"title":"After $20$ EM steps", "x":"$x$", "y":"$y$"})
    if i == 0:
        plt.title('Initialisation')
    plt.xticks([-2, 0, 2]), plt.yticks([-2, 0, 2])
    remove_axes(subplot = (1, 2, i + 1))
    axes.append(plt.gca())


ax0tr = axes[0].transData # Axis 0 -> Display
ax1tr = axes[1].transData # Axis 1 -> Display
figtr = fig.transFigure.inverted() # Display -> Figure
# 2. Transform arrow start point from axis 0 to figure coordinates
ptB = figtr.transform(ax0tr.transform((2.5, -0.2)))
# 3. Transform arrow end point from axis 1 to figure coordinates
ptE = figtr.transform(ax1tr.transform((-2, -0.2)))
# 4. Create the patch
arrow = matplotlib.patches.FancyArrowPatch(
    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
    fc = "orange", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha = 1,
    mutation_scale = 40.
)
# 5. Add patch to list of objects to draw onto the figure

fig.patches.append(arrow)

plt.tight_layout()
plt.show()
