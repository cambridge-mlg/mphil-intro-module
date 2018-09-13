import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def update(k):
    global frame
    global cb

    # calculate weights for this frame
    if frame >= 11:
        w1,w2 = [0,(frame - 10)]
    else:
        w2 = np.cos(1/2*np.pi*frame/11)
        w1 = np.sin(1/2*np.pi*frame/11)

    # calculate x matrix
    y = np.zeros([no_increments,no_increments])

    for i in range(no_increments):
        for j in range(no_increments):
            y[i][j] = 1/(1 + np.exp(-(w1*x1[j]) - (w2*x2[i])))

    # plot graphs
    ax1.clear()
    ax1.set_zlim(-0.05, 1.05)
    ax1.set_ylim(-5.05,5.05)
    ax1.set_xlim(-5.05,5.05)
    ax1.set_title("w = [" + str(round(w1,1)) + "," + str(round(w2,1)) + "]")
    X1,X2 = np.meshgrid(x1,x2)
    surf = ax1.plot_surface(X1, X2, y, rstride=1, cstride=1,
                cmap=cm.coolwarm, edgecolor='black', linewidths=0.2)

    ax2.clear()
    ax2.contourf(x1, x2, y, cmap = cm.coolwarm, alpha = 0.5)
    
    # add colorbar if not already added
    if not cb:
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb = True

    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')

    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('a')

    # increment frame number
    frame = (frame + 1)%16
    

# initialise figure and axis
fig = plt.figure(figsize=(10, 5))

# ax1 will be the surface plot and ax2 will be the contour plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
l, b, w, h = ax1.get_position().bounds
ax1.set_position([l - 0.03, b, w, h])
ax2 = fig.add_subplot(1, 2, 2)

# set the number of values plotted along each axis.
no_increments = 20

# initialise variables
x1 = np.linspace(-5,5,no_increments)
x2 = np.linspace(-5,5,no_increments)

frame = 0

cb = False

if __name__ == '__main__':
    # FuncAnimation calls update function for each frame
    anim = FuncAnimation(fig, update, frames=np.arange(0, 15), interval=500)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('single_neuron.gif', dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()