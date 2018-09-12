import numpy as np
import matplotlib.pyplot as plt

N = 40
classes = np.random.choice([0, 1], N)
no_1 = np.sum(classes)
no_2 = N - no_1

x_1 = np.random.normal(0, 1, no_1)
x_2 = np.random.normal(2, 1, no_2)

x = np.zeros(shape = (N, 1))
y = classes.reshape((-1,))

x[np.where(y == 0)[0], 0] = x_2
x[np.where(y == 1)[0], 0] = x_1

np.save('class_1d_inputs.npy', x)
np.save('class_1d_labels.npy', y)
