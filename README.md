# q2-sample-classifier
QIIME2 plugin for machine learning prediction of sample data.

This software is currently in pre-alpha phase and probably should not be used for anything other than testing.

If you use any of the code contained in this repository, please cite: https://github.com/nbokulich/q2-sample-classifier

## Examples
### Random forest classifier
```
cd ~/Desktop/projects/q2-sample-classifier/q2_sample_classifier/test_data/

qiime sample-classifier classify-random-forest --i-table feature-table-even11000-SedimentNoCrust-minfreq100mins5.qza --m-metadata-file glen-canyon-16S.tsv --p-category Type --o-visualization test --p-optimize-feature-selection --p-parameter-tuning --p-n-estimators 500
```