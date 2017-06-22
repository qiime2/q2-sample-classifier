# q2-sample-classifier

[![Build Status](https://travis-ci.org/nbokulich/q2-sample-classifier.svg?branch=master)](https://travis-ci.org/nbokulich/q2-sample-classifier) [![Coverage Status](https://coveralls.io/repos/github/nbokulich/q2-sample-classifier/badge.svg?branch=master)](https://coveralls.io/github/nbokulich/q2-sample-classifier?branch=master)

QIIME2 plugin for machine learning prediction of sample data.

This software is currently in pre-alpha phase and the API may be subject to change without notice.

If you use any of the code contained in this repository, please cite: https://github.com/nbokulich/q2-sample-classifier

This plugin requires QIIME2. See the [QIIME2 documentation](https://qiime2.org/) for installation and use of the QIIME2 framework.

Not sure which model to use? A good starting point is [this flowchart](http://scikit-learn.org/dev/tutorial/machine_learning_map/index.html). Most of the classification and regression models shown in that chart (and a few extras) are implemented in q2-sample-classifier.

# Quick links
[Installation](https://github.com/nbokulich/q2-sample-classifier#installation)
[Examples](https://github.com/nbokulich/q2-sample-classifier#examples)
* [Sample classification](https://github.com/nbokulich/q2-sample-classifier#classification)
* [Regression](https://github.com/nbokulich/q2-sample-classifier#regression)
* ["Maturity index" prediction](https://github.com/nbokulich/q2-sample-classifier#maturity-index-prediction)
* [Outlier detection](https://github.com/nbokulich/q2-sample-classifier#outlier-detection)
* [Predicting geospatial coordinates](https://github.com/nbokulich/q2-sample-classifier#predicting-geospatial-coordinates)
[Troubleshooting](https://github.com/nbokulich/q2-sample-classifier#troubleshooting)


# Installation
q2-sample-classifier requires the latest version of QIIME2. Install in conda using [these directions](https://docs.qiime2.org/2017.5/install/native/#install-qiime-2-within-a-conda-environment) and activate that conda environment. The install q2-sample-classifier with `pip` as follows:
```
pip install https://github.com/nbokulich/q2-sample-classifier/archive/master.zip
```


# Examples
In the examples below, any visualizations (QZV) files can be viewed in [view.qiime2.org](https://view.qiime2.org/). All example QZVs are located [here](.q2_sample_classifier/test_data/).

Supervised classification models predict sample data (e.g., metadata values) as a function of other sample data (e.g., microbiota composition). The predicted targets may be discrete sample classes (for classification problems) or continuous values (for regression problems). Any other data may be used as predictive features, but for the purposes of q2-sample-classifier this will most commonly be microbial sequence variant, operational taxonomic unit (OTU), or taxonomic composition. However, any features contained in a feature table may be used — for non-microbial data, just convert your observation tables to biom format and [import to qiime2](https://docs.qiime2.org/2017.5/tutorials/importing/#feature-table-data).

First, the estimator is trained using a subset of the data (training set) that determines which features (e.g., microbial sequence variants) are most predictive of each sample value or class (e.g., metadata categories). 

The estimator then predicts the value/class for each sample in a test set, i.e., a subset of samples held out for testing model accuracy. The true value/class of these samples may either be known, e.g., if we are testing how well microbiota composition predicts some discrete metadata class, or unknown, e.g., if we have developed a classification model to predict a patient's susceptibility to infection.

All machine-learning models implemented in q2-sample-classifier support the option to automatically tune hyperparameters (i.e., model parameters that affect performance), optimizing performance for each specific classification problem (at the expense of more computational time). Almost all actions also support the option to automatically select the most informative features (e.g., sequence variants), further optimizing estimator accuracy. This is a useful option, as most studies may observe hundreds or thousands of features but only a handful of these features may be predictive, and others may even decrease model accuracy. Knowing which features are predictive is obviously highly valuable, e.g., to discovering disease biomarkers. If enabled, feature selection is shown in the visualization as recursive feature elimination (RFE) plots, which show model accuracy as features are recursively eliminated over many iterations. Feature importance is ranked at each step, and the features that yield optimal predictive accuracy are extracted for final model training. All q2-sample-classifier classifier and regressor visualizations also show a list of the features used for prediction, and their importance, whether or not RFE optimization is enabled. Some methods, e.g., k-neighbors methods, do not rank feature importance and thus do not enable RFE, while several others, e.g., LASSO and ElasticNet, have a sort of built-in feature extraction method that can conflict with RFE, and hence are recommended to run without RFE optimization.

![Alt text](./examples/rfe_plot.jpg?raw=true "Recursive feature elimination plot")
## Classification
In a classification problem, we are interested in predicting the class labels for a number of unlabeled samples. The visualizers show overall accuracy scores (e.g., the percentage of times that the true class label is predicted), and also generates confusion matrices, showing how frequently samples belonging to each class are predicted to be the true class or another class. For example, the following depicts how frequently grape samples are classified to the correct Vineyard (94.1% of the time!):

![Alt text](./examples/classify-kneighbors-vineyard.jpg?raw=true "classify k-neighbors Vineyard")

```
cd ~/Desktop/projects/q2-sample-classifier/q2_sample_classifier/test_data/

qiime sample-classifier classify-samples \
	--i-table chardonnay.table.qza \
	--m-metadata-file chardonnay.map.txt \
	--p-category Vineyard \
	--o-visualization test \
	--p-optimize-feature-selection \
	--p-parameter-tuning \
	--p-n-estimators 50
```

## Regression
In a regression problem, we are interested in predicting numerical values for a number of unlabeled samples. The visualizers show overall accuracy scores (mean square error), and linear regression results (including R, P, and slope) for the predicted vs. expected test sample values. For example, the following depicts how accurately a random forests regressor predicts an infant's age as a function of it's stool microbiota composition:

![Alt text](./examples/age_regression.jpg?raw=true "Age regression")

```
qiime sample-classifier regress-samples \
	--i-table ecam-table-maturity.qza \
	--m-metadata-file ecam_map_maturity.txt \
	--p-category month \
	--o-visualization month \
	--p-optimize-feature-selection \
	--p-parameter-tuning \
	--p-n-estimators 50
```


## "Maturity Index" prediction

This method calculates a "microbial maturity" index from a regression model trained on feature data to predict a given continuous metadata category, e.g., to predict a subject's age as a function of microbiota composition. The model is trained on a subset of control group samples, then predicts the category value for all samples. This visualization computes maturity index z-scores (MAZ) to compare relative "maturity" between each group, as described in doi:10.1038/nature13421. This method can be used to predict between-group differences in relative trajectory across any type of continuous metadata gradient, e.g., intestinal microbiome development by age, microbial succession during wine fermentation, or microbial community differences along environmental gradients, as a function of two or more different "treatment" groups.

The visualizer produces a linear regression plot of predicted vs. expected values on the control test samples (as described above for regression models). Predicted vs. expected values are also shown for all samples in both control and test sets:

![Alt text](./examples/maz_predictions.jpg?raw=true "Age regression")

MAZ scores are calculated based on these predictions, statistically compared across all value "bin" (e.g., month of life) using ANOVA and paired t-tests, and shown as boxplots of MAZ distributions for each group in each value "bin" (e.g., month of life). A link within the visualizers allows download of the MAZ scores for each sample, facilitating customized follow-up testing, e.g., in R, **or use as metadata, e.g., for constructing PCoA plots**. In the maturity model shown below, we predicted infant age as a function of microbiota composition, and show that cesarean-born infants have lower MAZ scores at certain periods of development compared to vaginally born controls, indicating a slower trajectory of microbiota development.

![Alt text](./examples/maz_boxplots.jpg?raw=true "MAZ boxplots")

The average abundances of features used for training maturity models can then be viewed as heatmaps within the visualization. Feature abundance is averaged across all samples within each value bin (e.g., month of life) and within each individual sample group (e.g., vaginal controls vs. cesarean), demonstrating how different patterns of feature abundance (e.g., trajectories of development in the case of age or time-based models) may affect model predictions and MAZ scores.

![Alt text](./examples/maturity_heatmap.jpg?raw=true "Maturity heatmap")

All of this can be done with only a single command:

```
qiime sample-classifier maturity-index \
	--i-table ecam-table-maturity.qza \
	--m-metadata-file ecam_map_maturity.txt \
	--p-category month \
	--p-group-by delivery \
	--p-control Vaginal  \
	--p-n-jobs 4 \
	--o-visualization maturity \
	--p-test-size 0.4
```

## Outlier detection
This method detects contaminated samples and other outliers among your samples, tagging them for removal or follow-up study. Applications include but are not limited to detecting potentially contaminated samples, detecting potentially mislabeled samples, and detecting significant novelty, e.g., patients who responded to a treatment.

Input a feature table, possibly filtered to remove samples, depending on the goals of this analysis. Outliers can be detected from multiple sample types simultaneously, provided the goal is not to detect mislabeled samples or samples cross-contaminated with another sample type in this table. E.g., for detecting novelty or exogenous contaminants (e.g., from reagents), many different sample types may be tested simultaneously. Otherwise, the feature table should be filtered to contain only one or more sample classes between which cross-contamination is not suspected, or if these sample classes are highly resolved and mislabeled samples are not suspected. These assumptions may be supported by a preliminary principal coordinates analysis or other diversity analyses to determine how well resolved sample classes are and whether some sample classes appear to cluster with the wrong class(es).

Inputs support two different modes: if subset_category and subset_value are set, a subset of the input table is used as a "gold standard" sample pool for training the model. This mode is useful, for example, if you have a subset of "positive control" samples that represent the known diversity of your sample types (should be sufficiently large to capture normal extent of variation). Otherwise, the model is trained on all samples. Regardless of the input mode used, outlier status is predicted on all samples.

Returns a series of values documenting outlier status: inliers have value False, outliers have value True. This series may be input as a sample metadata file and used to filter a feature table, if appropriate, using q2_feature_table.filter_samples, to remove contaminants or focus on novelty samples. If interested in potentially mislabeled samples, use `classify_samples` or principal coordinates analysis to determine whether outliers classify as or cluster with another sample type.

```
qiime sample-classifier detect-outliers \
	--i-table chardonnay.table.qza \
	--m-metadata-file chardonnay.map.txt \
	--p-contamination 0.05 \
	--p-n-jobs 4 \
	--o-inliers outliers.qza
```
Let's view a PCoA plot of outliers vs. inliers:

![Alt text](./examples/outliers.jpg?raw=true "Outlier PCoA plot")
```
qiime feature-table rarefy \
	--i-table chardonnay.table.qza \
	--p-sampling-depth 2000 \
	--o-rarefied-table even_table
qiime diversity beta \
	--i-table even_table.qza \
	--o-distance-matrix  distance \
	--p-metric braycurtis
qiime diversity pcoa \
	--i-distance-matrix distance.qza \
	--o-pcoa  pcoa
qiime emperor plot \
	--i-pcoa  pcoa.qza \
	--o-visualization  outliers_plot \
	--m-metadata-file outliers.qza

```
We can then filter outliers from the feature table with the following command:
```
qiime feature-table filter-samples \
	--i-table chardonnay.table.qza \
	--o-filtered-table inliers-table \
	--m-metadata-file inliers.qza \
	--p-where "outlier='False'"
```

## Predicting geospatial coordinates
The method predict-coordinates allows us to predict two continuous variables on a single set of test data, allowing us to determine how well microbial composition predicts geographical source.
```
qiime sample-classifier predict-coordinates \
	--i-table chardonnay.table.qza \
	--m-metadata-file chardonnay.map.txt \
	--p-latitude latitude \
	--p-longitude longitude \
	--p-n-jobs 4 \
	--o-predictions coord-predictions \
	--o-prediction-regression coord-regression
```
This method generates a list of predicted latitude and longitude coordinates for each sample, contained in the 'predictions' artifact. The 'prediction-regression' contains linear regression and accuracy results for predicted vs. actual coordinates.

Furthermore, we can pass these results to the q2-coordinates plugin to visualize these results, mapping actual and predicted coordinates for each sample onto a map.(Note: this plugin is under LGPL license due to dependency requirements.)

![Alt text](./examples/predicted-coordinates.jpg?raw=true "Predicted coordinates plot")

```
qiime coordinates map-predicted-coordinates \
	--i-predictions coord-predictions.qza \
	--i-prediction-regression coord-regression.qza \
	--m-metadata-file chardonnay.map.txt \
	--p-latitude latitude \
	--p-longitude longitude \
	--p-pred-lat latitude \
	--p-pred-long longitude \
	--o-visualization prediction-map
```


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
