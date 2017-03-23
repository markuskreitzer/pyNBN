# Project 4
## Support Vector Regression

1.	Please download the LIBSVM package from www.eng.auburn.edu/~wilambm/ais/hand/SVMcode.zip and unpack it to a selected directory.  More information about the software you can find at: http://www.csie.ntu.edu.tw/~cjlin/libsvm/.

      Other useful information about SVM you may find at:
      * http://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf
      * http://svms.org/tutorials/Hearst-etal1998.pdf

      This package includes prepare_data.m script to generate 2000 training points and 1000 verification points for the MATLAB peaks function. Notice that both training and verification points are distributed randomly.

2.	Please modify the prepare_data.m script to generate 30*30 grid points for verification so you may then plot the resulted surface.

3.	Modify the SVR.m script so for each set of training parameters gamma and C (20 different plots) the resulted surface can be plotted. On figures print values of training and verifications errors.

4.	Run the SVR.m and  find the best values for gamma and C parameters using the simple grid search

5.	Based on results from (4) try to select try to reduce increments in the grid search so even better verification errors can be obtained.

6.	Plot the final surface and indicate resulted parameters such as number of RBF units, gamma, C, and errors. Also please inspect heights of RBF units (do not print them)


