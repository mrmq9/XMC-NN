
A PyTorch implementation of a simple neural network with one hidden layer for extreme multilable text classification with tf-idf representation.  
To tackle the problem of large output spaces, the loss can be computed using using uniform or in-batch negative sampling.

## Requirements

* python==3.9.7
* numpy==1.20.3
* scikit-learn==0.24.2
* pytorch==1.9.0

## Datasets

The datasets can be downloaded from the [extreme classification repository](http://manikvarma.org/downloads/XC/XMLRepository.html). Put the downloaded dataset in the "datasets" folder.   
The code only works with BoW features.  
Make sure that the training and evaluation data are named "train.txt" and "test.txt", respectively, before running the code.
