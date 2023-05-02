# coding=utf-8
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import copy

from lib.util import sse


class DctCompress(object):
    def __init__(self, z):
        self.sse_array = []
        self.z = z
        self.z_freq = dct(self.z, norm='ortho')

        # The line below creates a sorted list of all the energy levels in the DCT from low to high. Each item on the
        #  list contains a N-dimensional tuple with the coordinates to where the low energy element is in the DCT marix.
        self.sorted_indexes = np.dstack(np.unravel_index(np.argsort(np.square(self.z_freq.ravel())), z.shape))[0]
        self.total_waves = z.size
        self.used_indexes = copy.deepcopy(self.total_waves)

        # We want to give the user 5 updates of our progress
        # self.ord_num = int(np.floor(self.total_waves / 20))
        self.ord_num = 1000

    @property
    def compression_ratio(self):
        return float(self.used_indexes) / self.total_waves

    def compress(self, sse_max=0, target_compression_ratio=1.0, sse_history=False):
        """
        A generalized N-dimensional discrete cosine transform compression and data approximation algorithm. This
        algorithm is HORRIBLY slow. This is due to taking the inverse DCT after iteration as well as calculating SSE
        on each iteration. JPEG uses a "quality factor" which I need to research. Probably a better performance metric.
        """

        if sse_history:
            mse_array = []

        for coordinates in self.sorted_indexes:
            # Remove a cosine waveform.
            self.z_freq[tuple(coordinates)] = 0

            # Decrement index count and update compression ratio.
            # Compression ratio will be a between 0 and 1.
            self.used_indexes -= 1

            # Recover original.
            z_recovered = dct(self.z_freq, type=3, norm='ortho')

            # Calculate Sum-Squared-Error
            my_sse = sse(self.z, z_recovered)

            # if self.used_indexes % self.ord_num == 0:
            #     print("COMPRESSION: %f (TARGET: %f)" % (self.compression_ratio, target_compression_ratio))
            #     print("MSE: %s (max: %s)" % (my_sse, sse_max))
            #     print("Used_indexes %d (of %d)" % (self.used_indexes, self.total_waves))

            # This will of course slow down the function as well.
            if sse_history:
                self.sse_array.append(my_sse)

            if my_sse > sse_max and self.compression_ratio < target_compression_ratio:
                info = {'ratio': self.compression_ratio, 'num_indexes': self.used_indexes, 'sse': my_sse}
                if sse_history:
                    info['sse_array'] = self.sse_array
                return z_recovered, info


if __name__ == '__main__':
    """
    This is a example of how this works. Beware that this algorithm is not optimized. It will take very long to run.
    """
    import matplotlib.image as mpimg

    z = mpimg.imread('img/test.png')
    # fig = plt.figure()
    # plt.imshow(z)
    c = DctCompress(z)
    img, stats = c.compress(target_compression_ratio=0.9999, sse_history=True)

    # Plot Error
    fig = plt.figure()
    plt.title("SSE slope")
    plt.xlabel('iterations')
    plt.xlabel('SSE')
    plt.semilogy(stats['sse_array'])

    print "Compression Ratio: %0.5f, Number of Indexes: %d/%d, Sum Squared Error: %s" % (
        stats['ratio'], stats['num_indexes'], c.total_waves, str(stats['sse']))

    # Plot recovered image
    fig = plt.figure()
    plt.imshow(img)
    plt.show()
