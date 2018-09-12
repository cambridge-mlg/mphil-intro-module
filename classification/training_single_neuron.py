import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def sig(x):
    
    return 1/(1 + np.exp(-x)) # define logistic function for convenience

def gradient_ascent(x, y, init_weights, no_steps, stepsize): # x: train inputs, y: train labels, rest self explanatory
    
    x = np.append(np.ones(shape = (x.shape[0], 1)), x, axis = 1) # add 1's to the inputs as usual
    
    w = init_weights.copy() # copy weights (to prevent changing init_weights as a side-effect - don't dwell on this)
    
    w_history, log_liks = [], [] # arrays for storing weights and log-liklihoods at each step
    
    for n in range(no_steps): # in this part we optimise log-lik w.r.t. w
        
        log_liks.append(np.sum(y * np.log(sig(x.dot(w))) + (1 - y) * np.log(1 - sig(x.dot(w))))) # record current log-lik
        
        w_history.append(w.copy()) # record current weights (use w.copy() to prevent aliasing - don't dwell on this)
    
        sigs = sig(x.dot(w)) # using our neat convenience function
        
        dL_dw = np.mean((y - sigs)*x.T, axis = 1) # calculate gradient of log-likelihood w.r.t. w
        
        w += stepsize * dL_dw # update weights and repeat
    
    return np.array(w_history), np.array(log_liks) 

def update(k):
    global frame
    global cb
    global log_liks

    # calculate weights for this frame
    b,w1,w2 = w_history[frame]

    # calculate x matrix
    y = np.zeros([no_increments,no_increments])

    for i in range(no_increments):
        for j in range(no_increments):
            y[j][i] = 1/(1 + np.exp(-(w1*x1_axes[j]) - (w2*x2_axes[i]) - b))

    # plot graphs
    ax1.clear()
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('$p(y = 1|\mathbf{w}, x)$')
    ax1.set_zlim(-0.05, 1.05)
    ax1.set_xlim(x1_lims[0],x1_lims[1])
    ax1.set_ylim(x2_lims[0],x2_lims[1])
    ax1.set_title("w = [" + str(round(w1,2)) + "," + str(round(w2,2)) + "]")
    X1_axes,X2_axes = np.meshgrid(x1_axes,x2_axes)
    ax1.plot_wireframe(X1_axes, X2_axes, y, rstride=1, cstride=1)

    ax2.clear()
    ax2.set_xlabel('Step #')
    ax2.set_ylabel('Log-likelihood')
    ax2.set_xlim(0,no_frames)
    ax2.set_ylim(log_liks[0],0)
    ax2.plot(log_liks[:frame], color = 'black')

    ax3.clear()
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_xlim(x1_lims[0],x1_lims[1])
    ax3.set_ylim(x2_lims[0],x2_lims[1])
    cont = ax3.contourf(x1_axes, x2_axes, y, cmap = cm.coolwarm, alpha = 0.5)

    for c in range(2):
        X1,X2 = np.column_stack((x1,x2))[np.where(np.logical_not(y_data == c))[0]].transpose()
        ax1.scatter(X1,X2,[0]*len(X1),marker='x',color=['r','g'][c],alpha = 1)
        ax3.scatter(X1,X2,marker='x',color=['r','g'][c])

    # add colorbar if not already added
    if not cb:
        cb = fig.colorbar(cont, aspect=5)
        cb = True

    # increment frame number
    frame = (frame + 1)%no_frames

# initialise figure and axis
fig = plt.figure(figsize=(10, 5))

# ax1 will be the surface plot, ax2 will be the contour plot, and ax3 the log-likelihood
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
l, b, w, h = ax1.get_position().bounds
ax1.set_position([l- 0.05, b, w, h])
ax2 = fig.add_subplot(2, 2, 2) 
l, b, w, h = ax2.get_position().bounds
ax2.set_position([l, b + 0.05, w, h])
ax3 = fig.add_subplot(2, 2, 4)

# initialise variables
no_increments = 20
no_frames = 300

frame = 0

cb = False

x_data = np.load('iris_inputs_2d.npy')
y_data = np.load('iris_labels.npy')

x_data = x_data[np.where(np.logical_not(y_data == 2))[0]] 
y_data = y_data[np.where(np.logical_not(y_data == 2))[0]] # removing the datapoints of class 2

w_init = np.array([0.0,-2.0,2.0])

w_history, log_liks = gradient_ascent(x_data, y_data, w_init, no_frames, 0.2) # calling the gradient ascent function

x1,x2 = x_data.transpose()

x1_lims = [min(x1) - 0.2, max(x1) + 0.1]
x2_lims = [min(x2) - 0.1, max(x2) + 0.2] 

x1_axes = np.linspace(x1_lims[0],x1_lims[1],no_increments)
x2_axes = np.linspace(x2_lims[0],x2_lims[1],no_increments)


if __name__ == '__main__':
    # FuncAnimation calls update function for each frame
    anim = FuncAnimation(fig, update, frames=np.arange(0, no_frames), interval=60)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('training_single_neuron.gif', dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()