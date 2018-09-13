from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
set_notebook_preferences()

x = np.load('radioactive.npy')

plt.figure(figsize =(7, 1))
plot = plt.scatter(x, np.zeros_like(x), marker = 'x',
                   color = 'red', s = 100, zorder = 4, clip_on=False) # plot the decays as red crosses
beautify_plot({"x":"$x$"}) # from helper_functions for convenience
rect = patches.Rectangle((5, -0.55), 45, 1.1, linewidth = 0.5, edgecolor='black', facecolor='blue', alpha = 0.2)
plt.gca().add_patch(rect)
plt.xlim([0, 55])
plt.ylim([-0.5, 0.5]) # set y axis limits to be 0 and 10
plt.gca().axhline(y=0, color='k')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.gca().spines['bottom'].set_position('center')
plt.xticks(np.arange(0, 55, 10))
plt.tight_layout()
plt.show()
