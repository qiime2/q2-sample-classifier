# q2-sample-classifier

[![Build Status](https://travis-ci.org/qiime2/q2-sample-classifier.svg?branch=master)](https://travis-ci.org/qiime2/q2-sample-classifier) [![Coverage Status](https://coveralls.io/repos/github/qiime2/q2-sample-classifier/badge.svg?branch=master)](https://coveralls.io/github/qiime2/q2-sample-classifier?branch=master)

QIIME2 plugin for machine learning prediction of sample data.

If you use any of the code contained in this repository, please cite: https://github.com/qiime2/q2-sample-classifier

This plugin requires QIIME2. See the [QIIME2 documentation](https://qiime2.org/) for installation and use of the QIIME2 framework.

Not sure which model to use? A good starting point is [this flowchart](http://scikit-learn.org/dev/tutorial/machine_learning_map/index.html). Most of the classification and regression models shown in that chart (and a few extras) are implemented in q2-sample-classifier.

# Examples

Use examples are provided in the [QIIME2 tutorial documentation](https://docs.qiime2.org/2017.8/tutorials/sample-classifier/).

# Troubleshooting
Here follow some common errors and their solutions.
```
The test_size = 8 should be greater or equal to the number of classes = 12
```
This indicates that you have fewer test samples than the number of metadata classes (unique values in `category`). Increase parameter `test_size`. If you continue to get this error, you probably have either too few samples, or too few samples per class.
```
The number of observations cannot be determined on an empty distance matrix
```
Occasionally, a single feature will be chosen for model training, resulting in an empty distance matrix. This is probably not a useful model, anyway, so just run the command again and see if this changes. If you continue to get this error, you should examine your input data — ensure that you have a sufficient number of samples and many non-zero features. If you have very few samples or features (e.g., < 50), the maturity index model is probably not right for your data.
```
The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
```
The `Category` that you selected contains some classes with only one member. Remove these classes before continuing, or choose a better category!
```
Input contains NaN, infinity or a value too large for dtype('float64').
```
This error occurs most commonly if you have "NaN" values, e.g, empty rows, for a value you are trying to predict. Filter your sample metadata file to remove these samples before continuing!
