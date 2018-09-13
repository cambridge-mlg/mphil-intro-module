import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

fig = plt.figure(figsize=(9, 9))

# formatting main axis
big_ax = fig.add_subplot(111)
big_ax.spines['top'].set_color('none')
big_ax.spines['right'].set_color('none')
big_ax.tick_params(labelcolor='black', top=False, bottom=True, left=True, right=False)
plt.xticks([0.2,0.5,0.9],[-2,0,2])
plt.yticks([0.2,0.5,0.9],[-2,0,2])
big_ax.set_xlabel('w1')
big_ax.set_ylabel('w2')
no_increments = 20

# initialising z values
z1 = np.linspace(-5,5,no_increments)
z2 = np.linspace(-5,5,no_increments)

# iterating over each subplot and plotting each graph
for x_coord in range(1,4):
    for y_coord in range(1,4):

        # setting weight values for this subplot        
        w1 = -4 + 2 * x_coord
        w2 = -4 + 2 * y_coord

        # adding a subplot in appropriate location
        ax = fig.add_subplot(3, 3, (x_coord - 1) * 3 + y_coord, projection='3d')

        # calculating x for this subplot
        x = np.zeros([no_increments,no_increments])

        for i in range(no_increments):
            for j in range(no_increments):
                x[i][j] = 1/(1 + np.exp(-(w1*z1[j]) - (w2*z2[i])))

        # plotting the graph in this subplot
        ax.set_zlim(-0.05, 1.05)
        ax.set_ylim(-5.05,5.05)
        ax.set_xlim(-5.05,5.05)
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l + 0.02, b + 0.06, w, h])
        Z1,Z2 = np.meshgrid(z1,z2)
        surf = ax.plot_surface(Z1, Z2, x, rstride=1, cstride=1,
                    cmap=cm.coolwarm, edgecolor='black', linewidths=0.2)


plt.savefig('weight_space_single_neuron.svg')
