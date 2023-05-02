from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def scatter_surf3_orig(x, y, z):
    """Plot a surface plot from x,y,z scatter data."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tri = np.asarray(list(zip(x, y)))
    triang = mtri.Triangulation(x, y)
    ax.plot_trisurf(triang, z, cmap=plt.cm.Spectral)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

