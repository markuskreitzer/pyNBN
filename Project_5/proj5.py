# coding=utf-8
from scipy.fftpack import dct, idct
import numpy as np

from lib.dct_compress import DctCompress
from lib.util import sse
from lib.plotting.plot import Plot
import matplotlib.pyplot as plt


# Step 1: Generate function.
def gen_data(num_steps=30):
    x1 = np.linspace(-1, 1, num_steps)
    x2 = np.linspace(-1, 1, num_steps)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.multiply((0.3 - np.dot(1.8, X1) + np.dot(2.7, X1 ** 2)),
                    np.exp(-1 - np.dot(6, X2) - np.dot(9, X1 ** 2) - np.dot(9, X2 ** 2))) - np.multiply(
        (np.dot(0.6, X1) - np.dot(27, X1 ** 3) - np.dot(243, X2 ** 5)),
        np.exp(np.dot(- 9, X1 ** 2) - np.dot(9, X2 ** 2))) - np.dot(1 / 30, np.exp(
        - 1 - np.dot(6, X1) - np.dot(9, X1 ** 2) - np.dot(9, X2 ** 2)))
    return x1, x2, Z


grid_size = 60
print "Grid Size: {0}x{0}".format(grid_size)
x1, x2, Z = gen_data(grid_size)

# Step 2: Plot ideal representation.
p = Plot(x1, x2, Z).surf(save_fig_path='img/ideal_surface{0}x{0}.png'.format(grid_size))

# Step 3: Perform DCT
z_freq = dct(Z, norm='ortho')

# Step 4: Perform inverse-DCT
Y_recovered = idct(z_freq, norm='ortho')
ideal_mse = sse(Z, Y_recovered)
print "Ideal DCT", sse(Z, Y_recovered)

# Step 5: Rejection Threshold Calculation
# My implementation (probably what everyone is doing):
# 1) Start by rejecting the DCT waveform with the lowest energy.
#    1) Unravel NxN grid,
#    2) Sort from lowest to highest energy level
#    3) Create index array based on these.
#    4) Shrink window incrementally removing lowest value (see compress.py for details).
c = DctCompress(Z)
target_sse = 0.005
z_recovered, stats = c.compress(sse_max=target_sse, sse_history=True)
print "Compression Ratio: %0.5f, Number of Indexes: %d/%d, Sum Squared Error: %s" % (
    stats['ratio'], stats['num_indexes'], c.total_waves, str(stats['sse']))

p = Plot(x1, x2, z_recovered).surf(save_fig_path='img/recovered_surface{0}x{0}.png'.format(grid_size))

# Plot Error
fig = plt.figure()
plt.title("SSE Rate")
plt.xlabel('Cosines Used')
plt.ylabel('SSE')
iterations = len(stats['sse_array'])
x = np.arange(stats['num_indexes'], z_recovered.size)[::-1]
y = stats['sse_array']
plt.xlim(max(x), 1)
plt.grid()
plt.semilogy(x, y)
final_point = (stats['num_indexes'], stats['sse'])
plt.plot(stats['num_indexes'], stats['sse'], marker='o', markersize=5)
#plt.annotate('Final Point (%0.3f, %0.5s)' % final_point, xy=final_point, ha='right', xytext=final_point)
plt.savefig('img/mse_chart_{0}x{0}.png'.format(grid_size), bbox_inches='tight')




# Repeat the same problem using 60 x 60 grid and find the compression ratio for this case too.
