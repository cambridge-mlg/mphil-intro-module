import numpy as np
import matplotlib.pyplot as plt
np.random.seed(20)

cov = np.array([[1, 0],
              [0, 0.01]])

mu = np.array([0, 1])


rands = np.random.multivariate_normal(mu, cov, 100)
rands[:, 0] /= abs(rands[:, 0])**0.4
rands[:, 0] *= np.pi/(2*abs(rands[:, 0]).max())*1.3


x_1, y_1 = rands[:, 1]*np.cos(rands[:, 0]), rands[:, 1]*np.sin(rands[:, 0])
y_1 += 0.5
plt.scatter(x_1, y_1, color = 'red', marker = 'x', s = 5)

rands = np.random.multivariate_normal(mu, cov, 100)
rands[:, 0] /= abs(rands[:, 0])**0.4
rands[:, 0] *= np.pi/(2*abs(rands[:, 0]).max())*1.3
rands[:, 0] += np.pi

x_2, y_2 = rands[:, 1]*np.cos(rands[:, 0]), rands[:, 1]*np.sin(rands[:, 0])
y_2 -= 0.5
plt.scatter(x_2, y_2, color = 'blue', marker = 'x', s = 5)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()

x = np.stack([np.concatenate([x_1, x_2]), np.concatenate([y_1, y_2])], axis = -1)
y = np.array([0]*100 + [1]*100)

p = np.random.permutation(np.arange(200))
x, y = x[p], y[p]

x[:, 0] /= abs(x[:, 0]).max()
x[:, 1] /= abs(x[:, 1]).max()

np.save('class_crescent_inputs.npy', x)
np.save('class_crescent_labels.npy', y)
