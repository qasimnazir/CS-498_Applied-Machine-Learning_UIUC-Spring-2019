# Homework 3: Principal Component Analysis
 
https://courses.engr.illinois.edu/cs498aml/sp2019/homeworks/homework3.html

## Goal
The goal of this homework is to use PCA to smooth the noise in the provided data. 

The assignment can be done using any programming language. You may use a PCA package if you so choose but remember you need to understand what comes out of the package to get the homework right!

## Problems

At here, you will find a five noisy versions of the Iris dataset, and a noiseless version. For each of the 5 noisy data sets, you should compute the principal components in two ways. In the first, you will use the mean and covariance matrix of the noiseless dataset. In the second, you will use the mean and covariance of the respective noisy datasets. Based on these components, you should compute the mean squared error between the noiseless version of the dataset and each of a PCA representation using 0 (i.e. every data item is represented by the mean), 1, 2, 3, and 4 principal components. The mean squared error here should compute the sum of the squared errors over the features and compute the mean of this over the rows. For example, if the noiseless version has two rows [1,2,3,4] and [0,0,0,0] and the reconstructed version is [1,2,3,0] and [1,1,1,1] the MSE would be (16 + 4) / 2 = 10

You should produce:

- A csv file showing your numbers filled in a table set out as below, where "N" columns represents the components calculated via the noiseless dataset and the "c" columns of the noisy datasets.
- Example: The entry corresponding to Dataset I and 2N should contain the mean squared error between the noiseless version of the dataset and the PCA representation of Dataset I, using 2 principal components computed from the mean and covariance matrix of the noiseless dataset.
- Update (for clarity of instructions): In all cases you compare the reconstruction with the noiseless dataset to get the MSE. 
The first part, with "N" columns asks to reconstruct the noisy datasets using the PCs of the noiseless dataset. 
The second part, with "c" columns asks to reconstruct the noisy datasets using the PCs of the noisy dataset.
- A csv file containing your reconstruction of Dataset I ("dataI.csv"), expanded onto 2 principal components, where mean and principal components are computed from Dataset I.
