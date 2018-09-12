import numpy as np

np.random.seed(0)

x_min = 5
x_max = 50
lamda = 20

xs = []
while len(xs) < 10:
    rand = 1 - np.random.rand(1)
    x = -lamda*np.log(rand)
    if x > 5 and x < 50:
        xs.append(x)

xs = np.array(xs)
np.save('radioactive.npy', xs)
