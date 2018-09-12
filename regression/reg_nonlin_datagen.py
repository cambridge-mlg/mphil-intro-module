import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

epsilon = 0.1
no_total = 50
no_small_set = 10

#xs = np.linspace(0, 1, no_total)
xs = np.random.uniform(0, 1, no_total)

ys = np.sin(2*np.pi*xs) + epsilon*np.random.normal(0, 1, no_total)
ys += epsilon*np.random.normal(0, 1, no_total)

p = np.random.permutation(np.arange(len(xs)))
xs = np.array(xs)[p]
ys = np.array(ys)[p]

np.save('reg_nonlin_x.npy', xs[:no_small_set])
np.save('reg_nonlin_y.npy', ys[:no_small_set])

np.save('reg_nonlin_x_extended.npy', xs)
np.save('reg_nonlin_y_extended.npy', ys)
