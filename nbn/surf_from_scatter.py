import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Load the data
dataFile = 'peaks2000.dat'
data = np.loadtxt(dataFile)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create the Delaunay triangulation
tri = Delaunay(np.column_stack((x, y)))

# Create the surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap=plt.cm.Spectral)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

