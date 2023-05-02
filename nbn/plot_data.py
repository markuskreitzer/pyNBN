from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def scatter_surf3(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # create surface using Delaunay triangulation
    from scipy.spatial import Delaunay
    tri = Delaunay(np.column_stack((x, y)))
    ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', edgecolor='none')
    
    # set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Scatter Surface Plot')
    
    # show plot
    plt.show()

dataFile='peaks2000.dat';
#dataFile='abalone.dat';
data = np.loadtxt(dataFile)
x = data[:,0]
y = data[:,1]
z = data[:,2]
scatter_surf3(x, y, z)

