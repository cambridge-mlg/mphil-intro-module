import numpy as np
import matplotlib.pyplot as plt

names = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
inputs = []
outputs = []

for line in open('iris_data.txt', 'r'):

    split = ''.join(line.split()).split(',')

    inp = split[:-1]
    out = names[split[-1]]

    inputs.append(inp)
    outputs.append(out)

inputs = np.array(inputs).astype('float')
outputs = np.array(outputs).astype('int')

inputs += np.random.normal(0, 0.01, inputs.shape)

p = np.random.permutation(np.arange(outputs.shape[0]))

inputs = inputs[p]
outputs = outputs[p]

colors = np.array(['b', 'r', 'g'])
plt.figure(figsize = (6, 6))
for i in range(4):
    for j in range(4):

        if not(i == j):

            plt.subplot(4, 4, i*4 + (j + 1))
            idx = np.arange(inputs.shape[0])
            ins_i = inputs[:, i]
            ins_j = inputs[:, j]
            plt.scatter(ins_i, ins_j, s = 1, color = colors[outputs])
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig('iris_matrix.svg')

colors = np.array(['g', 'r', 'b'])
np.save('iris_inputs_2d.npy', inputs[:, 0:2])
np.save('iris_inputs_full.npy', inputs)
np.save('iris_labels.npy', outputs)


