import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax3 = plt.axes(projection='3d')

#定义三维数据
n = 1000
X = np.zeros((n,n))
Y = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        X[i,j] = j/n * (n-i)/n
        Y[i,j] = i/n

Z = 0.5*(X-1)*Y*(Y-1)

ax3.plot_surface(X,Y,Z,cmap='rainbow') 

ax3.set_xlabel(chr(958))
ax3.set_ylabel(chr(951))
ax3.set_zlabel('h')


plt.show()