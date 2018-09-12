import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Georgia'

xs = np.linspace(-6, 6, 100)
logit = 1/(1 + np.exp(-xs))
probit = norm.cdf(xs)
probit_scaled = norm.cdf(xs*(np.pi/8)**0.5)

plt.plot(xs, logit, 'blue', clip_on = False, zorder = 4, label = 'Logistic')
plt.plot(xs, probit, 'green', clip_on = False,
         zorder = 3, label = 'Probit ($\lambda^2 = 1$)')
plt.plot(xs, probit_scaled, 'red', clip_on = False,
         zorder = 3, label = 'Probit ($\lambda^2 = \pi/8$)')
plt.title("Logistic and Probit functions", fontsize = 18)
plt.xlabel("$x$", fontsize = 16)
plt.ylabel("$\sigma(x)$", fontsize = 16)
plt.xlim([-6, 6]), plt.ylim([0, 1])
plt.gca().legend(loc = 2, fontsize = 12)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.show()
