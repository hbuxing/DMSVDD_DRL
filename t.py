# 不同自由度的学生t分布与标准正态分布
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt

x = np.linspace( -5, 5, 100)
plt.plot(x, t.pdf(x,0.1), label='v=0.1')
plt.plot(x, t.pdf(x,1), label='v=1')
plt.plot(x, t.pdf(x,2), label = 'v=2')
plt.plot(x, t.pdf(x,100), label = 'v=100')
plt.plot(x, norm.pdf(x,0),'kx', label='normal')
plt.legend()
plt.show()