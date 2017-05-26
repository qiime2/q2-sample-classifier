# q2-sample-classifier
QIIME2 plugin for machine learning prediction of sample data.

This software is currently in pre-alpha phase and probably should not be used for anything other than testing.

If you use any of the code contained in this repository, please cite: https://github.com/nbokulich/q2-sample-classifier

See the [QIIME2 documentation](https://qiime2.org/) for installation and use of the QIIME2 framework.

Not sure which model to use? A good starting point is [this flowchart](http://scikit-learn.org/dev/tutorial/machine_learning_map/index.html). Most of the classification and regression models shown in that chart (and a few extras) are implemented in q2-sample-classifier.

# Examples
## Classification
### Random forest classifier
```
cd ~/Desktop/projects/q2-sample-classifier/q2_sample_classifier/test_data/

qiime sample-classifier classify-random-forest --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-category vineyard --o-visualization test --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Extra Trees classifier
```
qiime sample-classifier classify-extra-trees --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-category vineyard --o-visualization etc --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### AdaBoost classifier
```
qiime sample-classifier classify-adaboost --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-category vineyard --o-visualization abc --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Gradient boosting classifier
```
qiime sample-classifier classify-gradient-boosting --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-category vineyard --o-visualization gbc --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### K-nearest neighbors classifier
```
qiime sample-classifier classify-kneighbors --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-category vineyard --o-visualization knn --p-parameter-tuning
```
### Linear support vector machine classifier
```
qiime sample-classifier classify-linearSVC --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-category vineyard --o-visualization lsvc --p-parameter-tuning --p-optimize-feature-selection

qiime sample-classifier classify-SVC --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-category vineyard --o-visualization svcl --p-parameter-tuning --p-optimize-feature-selection --p-kernel linear
```
### Support vector machine classifier
```
qiime sample-classifier classify-SVC --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-category vineyard --o-visualization svc --p-parameter-tuning --p-optimize-feature-selection
```

## Regression
### Random forest regressor
```
qiime sample-classifier regress-random-forest --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization month --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Extra Trees regressor
```
qiime sample-classifier regress-extra-trees --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization etr --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### AdaBoost regressor
```
qiime sample-classifier regress-adaboost --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization abr --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Gradient boosting regressor
```
qiime sample-classifier regress-gradient-boosting --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization gbr --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Linear support vector machine regressor
```
qiime sample-classifier regress-SVR --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization svrl --p-parameter-tuning --p-optimize-feature-selection --p-kernel linear
```
### Support vector machine regressor
```
qiime sample-classifier regress-SVR --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization svr --p-parameter-tuning --p-optimize-feature-selection
```
### Ridge linear regression
```
qiime sample-classifier regress-ridge --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization ridge --p-parameter-tuning
```
### Lasso linear regression
```
qiime sample-classifier regress-lasso --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization lasso --p-parameter-tuning
```
### Elastic Net linear regression
```
qiime sample-classifier regress-elasticnet --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization elastic --p-parameter-tuning --p-optimize-feature-selection
```
### K-nearest neighbors regression
```
qiime sample-classifier regress-kneighbors --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --o-visualization knnr --p-parameter-tuning
```

## "Maturity Index" prediction

This method calculates a "microbial maturity" index from a regression model trained on feature data to predict a given continuous metadata category, e.g., to predict a subject's age as a function of microbiota composition. The model is trained on a subset of control group samples, then predicts the category value for all samples. This visualization computes maturity index z-scores to compare relative "maturity" between each group, as described in doi:10.1038/nature13421. This method can be used to predict between-group differences in relative trajectory across any type of continuous metadata gradient, e.g., intestinal microbiome development by age, microbial succession during wine fermentation, or microbial community differences along environmental gradients, as a function of two or more different "treatment" groups.

```
qiime sample-classifier maturity-index --i-table ecam-table-maturity.qza --m-metadata-file ecam_map_maturity.txt --p-category month --p-group-by delivery --p-control Vaginal  --p-n-jobs 4 --o-visualization maturity --p-test-size 0.4
```

## Outlier detection
This method detects contaminated samples and other outliers among your samples, tagging them for removal or follow-up study.

```
qiime sample-classifier detect-outliers --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-contamination 0.05 --p-n-jobs 4 --o-inliers inliers.qza
```
Let's view a PCoA plot of outliers vs. inliers:
```
qiime feature-table rarefy --i-table chardonnay.table.qza --p-sampling-depth 2000 --o-rarefied-table even_table
qiime diversity beta --i-table even_table.qza --o-distance-matrix  distance --p-metric braycurtis
qiime diversity pcoa --i-distance-matrix distance.qza --o-pcoa  pcoa
qiime emperor plot --i-pcoa  pcoa.qza --o-visualization  inliers_plot --m-metadata-file inliers.qza

```
We can then filter outliers from the feature table with the following command:
```
qiime feature-table filter-samples --i-table chardonnay.table.qza --o-filtered-table inliers-table --m-metadata-file inliers.qza --p-where "inlier='1'"
```

## Predicting geospatial coordinates
The method predict-coordinates allows us to predict two continuous variables on a single set of test data, allowing us to determine how well microbial composition predicts geographical source.
```
qiime sample-classifier predict-coordinates --i-table chardonnay.table.qza --m-metadata-file chardonnay.map.txt --p-latitude latitude --p-longitude longitude --p-n-jobs 4 --o-predictions coord-predictions --o-prediction-regression coord-regression
```
This method generates a list of predicted latitude and longitude coordinates for each sample, contained in the 'predictions' artifact. The 'accuracy' artifact contains accuracy scores for each coordinate, and 'prediction-regression' contains linear regression results for predicted vs. actual coordinates.

Furthermore, we can pass these results to the q2-coordinates plugin to visualize these results, mapping actual and predicted coordinates for each sample onto a map.(Note: this plugin is under LGPL license due to dependency requirements.)
```
qiime coordinates map-predicted-coordinates --i-predictions coord-predictions.qza --i-prediction-regression coord-regression.qza --m-metadata-file chardonnay.map.txt --p-latitude latitude --p-longitude longitude --p-pred-lat latitude --p-pred-long longitude --o-visualization prediction-map
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
