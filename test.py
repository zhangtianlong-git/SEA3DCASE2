import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection="3d")
Z = np.loadtxt('dem.txt')
x = np.arange(Z.shape[0])
y = np.arange(Z.shape[1])
X, Y = np.meshgrid(x, y)
ZNEW = (np.flip(Z, axis=0)).T
ax.plot_surface(X, Y, ZNEW, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
# ax.set_zlim(4450, 4580)
plt.show()
print('hi!!!')