# Homework 5: Vector quantization and classification
 
https://courses.engr.illinois.edu/cs498aml/sp2019/homeworks/homework5.html

## Goal
This homework focuses on vector quantization and classification. More specifically, you should do 1) data slicing,  2) vector clustering, 3) making histograms, and 4) building a multi-class classifer. We also encourage you to do in-depth experimentation and analysis.
The assignment can be done using any language. You may use packages for k-means, for nearest neighbors, and for whichever classification method you choose.

## Problems

Obtain the actitivities of daily life dataset from the UC Irvine machine learning website (https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer, data provided by Barbara Bruno, Fulvio Mastrogiovanni and Antonio Sgorbissa). Ignore the directories with MODEL in the name. They are duplicates.

(a) Build a classifier that classifies sequences into one of the 14 activities provided and evaluate its performance using average accuracy over 3 fold cross validation. To do the cross validation, divide the data for each class into 3 folds separately. Then, for a given run you will select 2 folds from each class for training and use the remaining fold from each class for test. To make features, you should vector quantize, then use a histogram of cluster center. This method is described in great detail in the book in section 9.3 which begins on page 166. You will find it helpful to use hierarchical k-means to vector quantize. You may perform the vector quantization for the entire dataset before doing cross validation.

You may use whatever multi-class classifier you wish, though we'd suggest you use a decision forest because it's easy to use and effective.

You should report (i) the average error rate over 3 fold cross validation and (ii) the class confusion matrix of your classifier for the fold with the lowest error, i.e. just one matrix for the 3 folds.

(b) Now see if you can improve your classifier by (i) modifying the number of cluster centers in your hierarchical k-means and (ii) modifying the size of the fixed length samples that you see.
