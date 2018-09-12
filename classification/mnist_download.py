import mnist
import numpy as np
import matplotlib.pyplot as plt

images = np.append(mnist.train_images(), mnist.test_images(), axis = 0)
labels = np.append(mnist.train_labels(), mnist.test_labels(), axis = 0)

np.save('mnist_images.npy', images)
np.save('mnist_labels.npy', labels)
