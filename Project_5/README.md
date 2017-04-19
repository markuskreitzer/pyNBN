**Advanced Intelligent Systems (ELEC 7970)**

**Project 5**

**Discrete Cosine Transform**

**DUE:  March 21**

1. Please generate data file for peaks function placing data points on regular 30 x 30 grid. To generate data points pleas use  the following equation:

        z3(j,i) =  (0.3-1.8\*x(i)+2.7\*x(i).^2).\*exp(-1-6\*y(j)-9\*x(i).^2-9\*y(j).^2) ...

            - (0.6\*x(i)-27\*x(i).^3-243\*y(j).^5).\*exp(-9\*x(i).^2-9\*y(j).^2) ...

            - 1/30\*exp(-1-6\*x(i)-9\*x(i).^2-9\*y(j).^2);

1. Plot Peaks function using the data file.

1. Using the data file perform DCT2 transformation into "frequency domain".

1. Perform IDCT2 transformation to obtain back original data points and plot error surface and calculate RMSE.

1. Adjust the rejection threshold in order to obtain the RMSE around 0.005 and find how many "frequency" values have to be used out of original 900.

1. Repeat the same problem using 60 x 60 grid and find the compression ratio for this case too.