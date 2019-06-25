# Homework 1: Classification With Naive Bayes

https://courses.engr.illinois.edu/cs498aml/sp2019/homeworks/homework1.html#about

## Goal

This homework focuses on classification using simple naive Bayes and decision forest classifiers. 
Additionally, it is intended to provide practice with finding and using publicly available libraries, an essential skill when applying machine learning techniques.

## Problems

### Problem 1: Diabetes Classification

A famous collection of data on whether a patient has diabetes, known as the Pima Indians dataset, and originally owned by the National Institute of Diabetes and Digestive and Kidney Diseases can be found at Kaggle. Download this dataset from https://www.kaggle.com/kumargh/pimaindiansdiabetescsv. This data has a set of attributes of patients, and a categorical variable telling whether the patient is diabetic or not. For several attributes in this data set, a value of 0 may indicate a missing value of the variable. There are a total of 767 data-points.

#### Part 1A 

Build a simple naive Bayes classifier to classify this data set. You should use a normal distribution to model each of the class-conditional distributions.

Compute an estimate of the accuracy of the classifier by averaging over 10 test-train splits. Each split should randomly assign 20% of the data to test, and the rest to train.
You should write this classifier and the test-train split code yourself (it's quite straight-forward).  Libraries can be used to load & hold the data.

#### Part 1B
Now adjust your code so that, for attribute 3 (Diastolic blood pressure), attribute 4 (Triceps skinfold thickness), attribute 6 (Body mass index), and attribute 8 (Age), it regards a value of 0 as a missing value when estimating the class-conditional distributions, and the posterior. 

Compute an estimate of the accuracy of the classifier by averaging over 10 test-train splits.

### Problem 2: MNIST Image Classification

The MNIST dataset is a dataset of 60,000 training and 10,000 test examples of handwritten digits, originally constructed by Yann Lecun, Corinna Cortes, and Christopher J.C. Burges. It is very widely used to check simple methods. There are 10 classes in total ("0" to "9"). This dataset has been extensively studied, and there is a history of methods and feature constructions at https://en.wikipedia.org/wiki/MNIST_database and at the original site, http://yann.lecun.com/exdb/mnist/. You should notice that the best methods perform extremely well.

The http://yann.lecun.com/exdb/mnist/ dataset is stored in an unusual format, described in detail on the page.  You do not have to write your own reader.  A web search should yield solutions for both Python and R.  For Python, https://pypi.org/project/python-mnist/ should work.  For R, there is reader code available at https://stackoverflow.com/questions/21521571/how-to-read-mnist-database-in-r. Please note that if you follow the recommendations in the accepted answer there at https://stackoverflow.com/a/21524980, you must also provide the readBin call with the flag signed=FALSE since the data values are stored as unsigned integers.

The dataset consists of 28 x 28 images. These were originally binary images, but appear to be grey level images as a result of some anti-aliasing. I will ignore mid-grey pixels (there aren't many of them) and call dark pixels "ink pixels", and light pixels "paper pixels"; you can modify the data values with a threshold to specify the distinction, as described here https://en.wikipedia.org/wiki/Thresholding_(image_processing). The digit has been centered in the image by centering the center of gravity of the image pixels, but as mentioned on the original site, this is probably not ideal. Here are some options for re-centering the digits that I will refer to in the exercises.

#### Part 2A: MNIST using naive Bayes

Model each class of the dataset using a Normal distribution and (separately) a Bernoulli distribution for both untouched images v. stretched bounding boxes, using 20 x 20 for your bounding box dimension.  This should result in 4 total models.  Use the training set to calculate the distribution parameters.

You must write the naive Bayes prediction code.  The distribution parameters can be calculated manually or via libraries.  Additionally, we recommend using a library to load the MNIST data (e.g. python-mnist or scikit-learn) and to rescale the images (e.g. openCV).

Compute the accuracy values for the four combinations of Normal v. Bernoulli distributions for both untouched images v. stretched bounding boxes.  Both the training and test set accuracy will be reported.
For each digit, plot the mean pixel values calculated for the Normal distribution of the untouched images.  In Python, a library such as matplotlib should prove useful.

#### Part 2B: MNIST using Decision Forest

Classify MNIST using a decision forest.
For your forest construction, you should investigate four cases. Your cases are: number of trees = (10, 30) X maximum depth = (4, 16). You should compute your accuracy for each of the following cases: untouched raw pixels; stretched bounding box. This yields a total of 8 slightly different classifiers. Please use 20 x 20 for your bounding box dimensions.
You should use a decision forest library.  No need to write your own.
