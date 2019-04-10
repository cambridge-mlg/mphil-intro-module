import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

x = np.load('clustering_2d.npy')
rs = np.load('clustering_2d_classes.npy').astype(int)
print(rs)
colors = np.array(['red', 'green', 'blue'])

fig = plt.figure(figsize = (8, 5))
plt.subplot(121)
plt.scatter(x[:, 0], x[:, 1], marker = 'o', color = 'white', edgecolor = 'black')
ax_0 = plt.gca()
ax_0.set_aspect('equal')
plt.xlim([-3, 3]), plt.ylim([-2.5, 2.5])
remove_axes()


plt.subplot(122)
from matplotlib.patches import Ellipse

ells = [Ellipse((-0.7, -1.4), 4, 1.6, 20,
               edgecolor='black', lw=2, facecolor='none'),
       Ellipse((0.9, 0.5), 2.7, 1.6, -15,
               edgecolor='black', lw=2, facecolor='none'),
       Ellipse((-1.25, 1.1), 2.5, 0.75, 25,
               edgecolor='black', lw=2, facecolor='none')]

a = plt.subplot(122, aspect='equal')

for e in ells:
    e.set_clip_box(a.bbox)
    a.add_artist(e)

plt.scatter(x[:, 0], x[:, 1], marker = 'o', color = colors[rs], edgecolor = 'black')

plt.xlim([-3.25, 2.75]), plt.ylim([-2.75, 2.25])
remove_axes()

ax_1 = plt.gca()

ax0tr = ax_0.transData # Axis 0 -> Display
ax1tr = ax_1.transData # Axis 1 -> Display
figtr = fig.transFigure.inverted() # Display -> Figure
# 2. Transform arrow start point from axis 0 to figure coordinates
ptB = figtr.transform(ax0tr.transform((2.5, -0.2)))
# 3. Transform arrow end point from axis 1 to figure coordinates
ptE = figtr.transform(ax1tr.transform((-2.5, -0.2)))
# 4. Create the patch
arrow = matplotlib.patches.FancyArrowPatch(
    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
    fc = "orange", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha = 1,
    mutation_scale = 40.
)
# 5. Add patch to list of objects to draw onto the figure

fig.patches.append(arrow)

plt.tight_layout()
plt.savefig('clust_example.svg', bbox_inches='tight')
plt.show()
