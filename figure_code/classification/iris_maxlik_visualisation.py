import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Georgia'

def sig(x):
    return 1/(1 + np.exp(-x))

def gradient_ascent_opt(x, y, init_weights, no_steps, stepsize):
    x = np.append(np.ones(shape = (x.shape[0], 1)), x, axis = 1)
    w = init_weights.copy()
    w_history = []
    log_liks = []
    count = 0
    res = 10
    x1 = np.linspace(4, 7.2, res + 1)
    x2 = np.linspace(1.9, 4.5, res + 2)
    x3 = np.linspace(0.5, 5.5, res + 3)
    x4 = np.linspace(0, 4, res + 4)
    grid = np.stack(np.meshgrid(x1, x2, x3, x4, indexing = 'ij'), axis = -1)
    grid = np.append(np.ones(shape = grid.shape[:-1] + (1,)), grid, axis = -1)
    axes = (x1, x2, x3, x4)
    
    for n in range(no_steps):
        plt.figure(figsize = (6, 6))
        for i in range(1, 5):
            for j in range(1, 5):
                k = (i - 1)*4 + j

                if not(i == j):
                    
                    plt.subplot(4, 4, k)
                    
                    sigs = sig(grid.dot(w))
                    idx = [0, 1, 2, 3]
                    idx.remove(i-1)
                    idx.remove(j-1)
                    values = sigs.mean(axis = tuple(idx))
                    if i < j:
                        values = values.T
                        
                    plt.contourf(axes[i-1], axes[j-1], values,
                                 cmap = 'coolwarm', alpha = 0.5)

                    
                    idx = np.arange(x.shape[0])
                    ins_i = x[:, i]
                    ins_j = x[:, j]
                    plt.scatter(ins_i, ins_j, marker = 'x', s = 1, color = np.array(['b', 'r'])[y])
                    frame = plt.gca()
                    title = '$w_{'+str(i)+'} = '+'{0:.{1}f}'.format(w[i], 2)+', w_{'+str(j)+'} = '+'{0:.{1}f}'.format(w[j], 2)+ "$"
                    plt.title(title, fontsize = 8)
                    frame.axes.get_xaxis().set_visible(False)
                    frame.axes.get_yaxis().set_visible(False)
                    

        plt.tight_layout()
        plt.savefig('{}.png'.format(str(n).zfill(3)), dpi = 400)
        plt.close()
        
        log_liks.append(np.sum(y*np.log(sig(x.dot(w))) + (1 - y)*np.log(1 - sig(x.dot(w)))))
        w_history.append(w.copy())
    
        sigs = sig(x.dot(w))
        dL_dw = np.mean((y - sigs)*x.T, axis = 1)
        w += stepsize*dL_dw
        count += 1

    w_history = np.array(w_history)
    log_liks = np.array(log_liks)
    w = w_history[-1]
    
    return w_history, log_liks, count

x = np.load('iris_inputs_full.npy')
y = np.load('iris_labels.npy')
os.chdir('class_iris_imgs')

x = x[np.where(np.logical_not(y == 2))[0]]
y = y[np.where(np.logical_not(y == 2))[0]]

w_history, log_liks, count = gradient_ascent_opt(x, y, np.random.normal(0, 1, 5), 200, 0.1)
