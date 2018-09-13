import numpy as np
import matplotlib.pyplot as plt

images = np.load('mnist_images.npy')
labels = np.load('mnist_labels.npy')

plt.figure(figsize = (6, 6))
for n in range(10):
    for j in range(10):

        plt.subplot(10, 10, n*10 + j + 1)

        idx = np.where(labels == n)[0][j+100]
        plt.imshow(images[idx, :, :], cmap = 'binary')
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        frame.axis('off')
        
plt.tight_layout()
plt.show()
