import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
epsilon = 0.1

m = 1.2
c = 0.3
no_points = 10

#xs = np.linspace(0, 1, no_points)
xs = np.random.uniform(0, 1, no_points)
ys = m*xs + c + epsilon*np.random.normal(0, 1, no_points)

xs = np.array(xs)
ys = np.array(ys)

p = np.random.permutation(xs.shape[0])
xs = xs[p]
ys = ys[p]

np.save('reg_lin_x.npy', xs)
np.save('reg_lin_y.npy', ys)
