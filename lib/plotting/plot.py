# coding=utf-8
import numpy as np
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Plot(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        self.z_min = np.min(z)
        self.x_min = np.min(x)
        self.x_max = np.max(x)

        self.y_min = np.min(y)
        self.y_max = np.max(y)

        self.z_min = np.min(z)
        self.z_max = np.max(z)
        self.fig = plt.figure()

    def _interpolate_grid(self):
        """

        :return:
        """
        # Below I try to reduce the size of the meshgrid that will be constructed by dividing by 10. Initially,
        self.xi = np.linspace(self.x_min, self.x_max, np.floor(np.sqrt(self.x.shape[0])))
        self.yi = np.linspace(self.y_min, self.y_max, np.floor(np.sqrt(self.y.shape[0])))

        # Put the data in a grid using linear triangulation to interpolate each point.
        self.zi = griddata(self.x, self.y, self.z, self.xi, self.yi, interp='linear')

    def surf_rand(self, data_points=False, save_fig_path=None, *args, **kwargs):
        """
        Surface plot the gridded data, plotting dots at the nonuniform data points as well.
        """
        self._interpolate_grid()
        ax = plt.gca(projection='3d')
        my_x, my_y = np.meshgrid(self.xi, self.yi)
        surf = ax.plot_surface(my_x, my_y, self.zi, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.01, antialiased=False)
        # Draw the colorbar
        # plt.colorbar()
        plt.colorbar(surf, shrink=0.5, aspect=5, extendfrac=True)

        # Plot the random data points.
        if data_points:
            plt.scatter(self.x, self.y, s=5, zorder=10)
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        if kwargs['title']:
            plt.title(kwargs['title'])

        if save_fig_path:
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close(self.fig)
        else:
            plt.show()

    def contour_rand(self, data_points=False, save_fig_path=None, *args, **kwargs):
        """
        Contour the gridded data, plotting dots at the nonuniform data points as well.
        """
        self._interpolate_grid()
        CS = plt.contour(self.xi, self.yi, self.zi, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(self.xi, self.yi, self.zi, 15, vmax=abs(self.zi).max(), vmin=-abs(self.zi).max())

        # Draw the colorbar
        # plt.colorbar()

        # Plot the random data points.
        if data_points:
            plt.scatter(self.x, self.y, s=5, zorder=10)

        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)

        if kwargs['title']:
            plt.title(kwargs['title'])

        if save_fig_path:
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close(self.fig)
        else:
            plt.show()

    def surf(self, save_fig_path=None, *args, **kwargs):
        ax = plt.gca(projection='3d')
        x, y = np.meshgrid(self.x, self.y)
        surf = ax.plot_surface(x, y, self.z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1, antialiased=False)
        z_min = np.min(self.z)
        z_max = np.max(self.z)
        ax.set_zlim(z_min, z_max)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        plt.colorbar(surf, shrink=0.5, aspect=5, extendfrac=True)

        if save_fig_path:
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close(self.fig)
        else:
            plt.show()

    def scatter(self, save_fig_path=None, *args, **kwargs):

        ax = plt.gca(projection='3d')
        ax.scatter(self.x, self.y, self.z, s=5, colors='k')

        ax.set_zlim(self.z_min, self.z_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # fig.colorbar(scat, shrink=0.5, aspect=5, extendfrac=True)

        if save_fig_path:
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close(self.fig)
        else:
            plt.show()

