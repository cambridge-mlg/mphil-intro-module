import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../..')
from helper_functions import *
from scipy.interpolate import splprep, splev
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Georgia'

matplotlib.rc('axes', titlesize = 18)
matplotlib.rc('axes', labelsize = 16)
matplotlib.rc('xtick', labelsize = 12)
matplotlib.rc('ytick', labelsize = 12)


x_lin = np.load('reg_lin_x.npy')
y_lin = np.load('reg_lin_y.npy')
x_lin = np.stack([np.ones(shape = x_lin.shape), x_lin], axis = 1)

w0 = np.linspace(-0.7, 1.3, 150)
w1 = np.linspace(0, 2, 100)
w_grid = np.stack(np.meshgrid(w0, w1), axis = -1)
error = np.sum((y_lin - (w_grid).dot(x_lin.T))**2, axis = -1)

w0_ = np.array([0.3, 0.55, 0.96, 1.14, 0.31, -0.04, 0.3])
w1_ = np.array([1.17, 1.4, 1.5, 1.23, 0.27, 0.24, 1.17])

tck, u = splprep(np.stack([w0_, w1_], axis = 0), u=None, s=0.0, per=1) 
u_new = np.linspace(u.min(), u.max(), 100)
w0s, w1s = splev(u_new, tck, der=0)


for i in range(15):
    w0_, w1_ = w0s[i], w1s[i]
    
    plt.figure(figsize = (8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, 1, 100), w0_ + w1_*np.linspace(0, 1, 100), color = 'black')
    beautify_plot({"title":"Data space", "x":"$x$", "y":"$y$"})
    plt.ylim([-0.5, 2.5])
    for j in range(x_lin.shape[0]):
        plt.plot([x_lin[j, 1], x_lin[j, 1]],
                 [y_lin[j], x_lin[j, 1]*w1_ + w0_], '--',
                 color = 'gray', zorder = 1)
    plt.scatter(x_lin[:, 1], y_lin, marker = 'x', color = 'red', zorder = 2)

    plt.subplot(1, 2, 2)
    plt.contourf(w0, w1, np.log(error), cmap = 'coolwarm', alpha = 0.5)
    plt.plot(w0s, w1s, color = 'black')
    plt.scatter(w0_, w1_, marker = 'x', color = 'black')
    beautify_plot({"title":"ln($E_2$) in weight space", "x":"$w_0$", "y":"$w_1$"})
    plt.yticks(np.linspace(0, 2, 5))
    plt.tight_layout()
    plt.savefig('{}.png'.format(str(i).zfill(2)), dpi = 400)
    plt.close()
print('done')

