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

qiime sample-classifier classify-random-forest --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization test --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Extra Trees classifier
```
qiime sample-classifier classify-extra-trees --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization etc --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### AdaBoost classifier
```
qiime sample-classifier classify-adaboost --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization abc --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Gradient boosting classifier
```
qiime sample-classifier classify-gradient-boosting --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization gbc --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### K-nearest neighbors classifier
```
qiime sample-classifier classify-kneighbors --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization knn --p-parameter-tuning
```
### Linear support vector machine classifier
```
qiime sample-classifier classify-linearSVC --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization lsvc --p-parameter-tuning --p-optimize-feature-selection

qiime sample-classifier classify-SVC --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization svcl --p-parameter-tuning --p-optimize-feature-selection --p-kernel linear
```
### Support vector machine classifier
```
qiime sample-classifier classify-SVC --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization svc --p-parameter-tuning --p-optimize-feature-selection
```

## Regression
### Random forest regressor
```
qiime sample-classifier regress-random-forest --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization estimated_elevation --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Extra Trees regressor
```
qiime sample-classifier regress-extra-trees --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization etr --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### AdaBoost regressor
```
qiime sample-classifier regress-adaboost --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization abr --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Gradient boosting regressor
```
qiime sample-classifier regress-gradient-boosting --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization gbr --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 50
```
### Linear support vector machine regressor
```
qiime sample-classifier regress-SVR --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization svrl --p-parameter-tuning --p-optimize-feature-selection --p-kernel linear
```
### Support vector machine regressor
```
qiime sample-classifier regress-SVR --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization svr --p-parameter-tuning --p-optimize-feature-selection
```
### Ridge linear regression
```
qiime sample-classifier regress-ridge --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization ridge --p-parameter-tuning
```
### Lasso linear regression
```
qiime sample-classifier regress-lasso --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization lasso --p-parameter-tuning
```
### Elastic Net linear regression
```
qiime sample-classifier regress-elasticnet --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization elastic --p-parameter-tuning --p-optimize-feature-selection
```
### K-nearest neighbors regression
```
qiime sample-classifier regress-kneighbors --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization knnr --p-parameter-tuning
```

## "Maturity Index" prediction

This method calculates a "microbial maturity" index from a regression model trained on feature data to predict a given continuous metadata category, e.g., to predict a subject's age as a function of microbiota composition. The model is trained on a subset of control group samples, then predicts the category value for all samples. This visualization computes maturity index z-scores to compare relative "maturity" between each group, as described in doi:10.1038/nature13421. This method can be used to predict between-group differences in relative trajectory across any type of continuous metadata gradient, e.g., intestinal microbiome development by age, microbial succession during wine fermentation, or microbial community differences along environmental gradients, as a function of two or more different "treatment" groups.

```
qiime sample-classifier maturity-index --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --p-group-by Site_Name --p-control Hcanyon --p-n-jobs 4 --o-visualization maturity --p-test-size 0.4
```

# Troubleshooting
Here follow some common errors and their solutions.
```The test_size = 8 should be greater or equal to the number of classes = 12```
This indicates that you have fewer test samples than the number of metadata classes (unique values in `category`). Increase parameter `test_size`. If you continue to get this error, you probably have either too few samples, or too few samples per class.
