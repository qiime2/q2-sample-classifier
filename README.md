# q2-sample-classifier
QIIME2 plugin for machine learning prediction of sample data.

This software is currently in pre-alpha phase and probably should not be used for anything other than testing.

If you use any of the code contained in this repository, please cite: https://github.com/nbokulich/q2-sample-classifier

# Examples
## Classification
### Random forest classifier
```
cd ~/Desktop/projects/q2-sample-classifier/q2_sample_classifier/test_data/

qiime sample-classifier classify-random-forest --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization test --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 500
```
### K-nearest neighbors classifier
```
qiime sample-classifier classify-kneighbors --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization knn --p-parameter-tuning
```
### Linear support vector machine classifier
```
qiime sample-classifier classify-linearSVC --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization svc --p-parameter-tuning

qiime sample-classifier classify-SVC --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization svc --p-parameter-tuning --p-kernel linear
```
### Support vector machine classifier
```
qiime sample-classifier classify-SVC --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization svc --p-parameter-tuning
```

## Regression
### Random forest regressor
```
qiime sample-classifier regress-random-forest --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization estimated_elevation --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 500
```
### Linear support vector machine regressor
```
qiime sample-classifier regress-SVR --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization svr --p-parameter-tuning --p-kernel linear
```
### Support vector machine regressor
```
qiime sample-classifier regress-SVR --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization svr --p-parameter-tuning
```
### Ridge linear regression
```
qiime sample-classifier regress-ridge --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization ridge --p-parameter-tuning
```
### Lasso linear regression
```
qiime sample-classifier regress-lasso --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category estimated_elevation --o-visualization lasso --p-parameter-tuning
```
