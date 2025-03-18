# Logistic-Regression-by-use-of-PCA
This kernel is all about Principal Component Analysis - a Dimensionality Reduction technique.

I have discussed Principal Component Analysis (PCA). In particular, I have introduced PCA, explained variance ratio, Logistic Regression with PCA, find right number of dimensions and plotting explained variance ratio with number of dimensions.

I have used the adult data set for this kernel. This dataset is very small for PCA purpose. My main purpose is to demonstrate PCA implementation with this dataset.

I hope you find this kernel useful and your UPVOTES would be very much appreciated

Table of Contents
The contents of this kernel is divided into various topics which are as follows:-

The Curse of Dimensionality
Introduction to Principal Component Analysis
Import Python libraries
Import dataset
Exploratory data analysis
Split data into training and test set
Feature engineering
Feature scaling
Logistic regression model with all features
Logistic Regression with PCA
Select right number of dimensions
Plot explained variance ratio with number of dimensions
Conclusion
References
The Curse of Dimensionality
Generally, real world datasets contain thousands or millions of features to train for. This is very time consuming task as this makes training extremely slow. In such cases, it is very difficult to find a good solution. This problem is often referred to as the curse of dimensionality.

The curse of dimensionality refers to various phenomena that arise when we analyze and organize data in high dimensional spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional settings. The problem is that when the dimensionality increases, the volume of the space increases so fast that the available data become sparse. This sparsity is problematic for any method that requires statistical significance.

In real-world problems, it is often possible to reduce the number of dimensions considerably. This process is called dimensionality reduction. It refers to the process of reducing the number of dimensions under consideration by obtaining a set of principal variables. It helps to speed up training and is also extremely useful for data visualization.

The most popular dimensionality reduction technique is Principal Component Analysis (PCA), which is discussed below.

Introduction to Principal Component Analysis (PCA)
Principal Component Analysis (PCA) is a dimensionality reduction technique that can be used to reduce a larger set of feature variables into a smaller set that still contains most of the variance in the larger set.

Preserve the variance
PCA, first identifies the hyperplane that lies closest to the data and then it projects the data onto it. Before, we can project the training set onto a lower-dimensional hyperplane, we need to select the right hyperplane. The projection can be done in such a way so as to preserve the maximum variance. This is the idea behind PCA.

Principal Components
PCA identifies the axes that accounts for the maximum amount of cumulative sum of variance in the training set. These are called Principal Components. PCA assumes that the dataset is centered around the origin. Scikit-Learn’s PCA classes take care of centering the data automatically.

Projecting down to d Dimensions
Once, we have identified all the principal components, we can reduce the dimensionality of the dataset down to d dimensions by projecting it onto the hyperplane defined by the first d principal components. This ensures that the projection will preserve as much variance as possible.
