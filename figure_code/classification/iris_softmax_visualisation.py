import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Georgia'

def softmax(x):
    if len(x.shape) == 2:
        return (np.exp(x).T/np.sum(np.exp(x), axis = -1)).T # define softmax function for convenience
    else:
        numerator = x.T
        return (np.exp(x).T/np.sum(np.exp(x), axis = -1).T).T

    
def softmax_gradient_ascent(x, y, init_weights, no_steps, stepsize, y_col):
    x = np.append(np.ones(shape = (x.shape[0], 1)), x, axis = 1)
    w = init_weights.copy()
    w_history = []
    log_liks = []
    count = 0
    res = 20
    x1 = np.linspace(4, 9, res + 1)
    x2 = np.linspace(1.9, 4.5, res + 2)
    x3 = np.linspace(0.5, 8, res + 3)
    x4 = np.linspace(0, 4, res + 4)
    grid = np.stack(np.meshgrid(x1, x2, x3, x4, indexing = 'ij'), axis = -1)
    grid = np.append(np.ones(shape = grid.shape[:-1] + (1,)), grid, axis = -1)
    axes = (x1, x2, x3, x4)
    
    for n in range(no_steps):
        log_liks.append(np.sum(y*np.log(softmax(x.dot(w)))))
        if n % 10 == 0:
            plt.figure(figsize = (6, 6))
            for i in range(1, 5):
                for j in range(1, 5):
                    k = (i - 1)*4 + j

                    if not(i == j):
                        
                        plt.subplot(4, 4, k)
                        probs = softmax(grid.dot(w))
                        rgb_colors = np.roll(probs, -1, axis  = -1)
                        idx = [0, 1, 2, 3]
                        idx.remove(i-1)
                        idx.remove(j-1)
                        values = rgb_colors.mean(axis = tuple(idx))
                        
                        if i < j:
                            values = np.swapaxes(values, 0, 1)
                        plt.imshow(values, alpha = 0.5, origin = 'lower',
                                   extent = [axes[i-1].min(), axes[i-1].max(),
                                             axes[j-1].min(), axes[j-1].max()],
                                   aspect = (axes[i-1].max() - axes[i-1].min())/(axes[j-1].max() - axes[j-1].min()))
                        
                        idx = np.arange(x.shape[0])
                        ins_i = x[:, i]
                        ins_j = x[:, j]
                        
                        plt.scatter(ins_i, ins_j, marker = 'x', s = 1, color = np.array(['b', 'r', 'g'])[y_col])
                        frame = plt.gca()
                        #title = '$w_{'+str(i)+'} = '+'{0:.{1}f}'.format(w[i], 2)+', w_{'+str(j)+'} = '+'{0:.{1}f}'.format(w[j], 2)+ "$"
                        plt.title("", fontsize = 8)
                        frame.axes.get_xaxis().set_visible(False)
                        frame.axes.get_yaxis().set_visible(False)
                        

            plt.tight_layout()
            plt.savefig('softmax_{}.png'.format(str(n).zfill(3)), dpi = 400)
            plt.close()
            print(n, log_liks[-1])
        log_liks.append(np.sum(y*np.log(softmax(x.dot(w))))) # record current log-lik as before
        w_history.append(w.copy()) # record current weights as before
    
        soft_ = softmax(x.dot(w)) # using our neat convenience function
        dL_dw = (x.T).dot(y - soft_)/x.shape[0]
        w += stepsize*dL_dw # update weights and repeat

##def softmax_gradient_ascent(x, y, init_weights, no_steps, stepsize):
##    x = np.append(np.ones(shape = (x.shape[0], 1)), x, axis = 1) # add 1's to the inputs as before
##    w = init_weights.copy() # copy weights as before
##    w_history, log_liks = [], [] # arrays for storing weights and log-liklihoods as before
##
##    for n in range(no_steps): # in this part we optimise log-lik w.r.t. ws
##        log_liks.append(np.sum(y*np.log(softmax(x.dot(w))))) # record current log-lik as before
##        w_history.append(w.copy()) # record current weights as before
##    
##        soft_ = softmax(x.dot(w)) # using our neat convenience function
##        dL_dw = (x.T).dot(y - soft_)/x.shape[0]
##        w += stepsize*dL_dw # update weights and repeat
##    
##    return np.array(w_history), np.array(log_liks)

x = np.load('iris_inputs_full.npy')
y = np.load('iris_labels.npy')
os.chdir('iris_class_imgs')

no_classes = 3
y_ = np.zeros(shape = (y.shape[0], no_classes))
y_[np.arange(y.shape[0]), y] = 1

no_train = (x.shape[0]*3)//4
x_train, x_test, y_train, y_test = x[:no_train], x[no_train:], y_[:no_train], y_[no_train:]

init_weights = np.random.normal(0, 0.1, (x_train.shape[1] + 1, no_classes))
w_history, log_liks = softmax_gradient_ascent(x_train, y_train, init_weights, 10**4, 0.1, y)
plt.plot(log_liks)
plt.show()
