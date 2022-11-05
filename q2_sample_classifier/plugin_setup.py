# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import importlib

from qiime2.plugin import (
    Int, Str, Float, Range, Bool, Plugin, Metadata, Choices, MetadataColumn,
    Numeric, Categorical, Citations, Visualization, TypeMatch)
from q2_types.feature_table import (
    FeatureTable, Frequency, RelativeFrequency, PresenceAbsence, Balance,
    PercentileNormalized, Design)
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData
from q2_types.distance_matrix import DistanceMatrix
from q2_feature_table import heatmap_choices
from .classify import (
    classify_samples, classify_samples_from_dist, regress_samples,
    regress_samples_ncv,
    classify_samples_ncv, fit_classifier, fit_regressor, split_table,
    predict_classification, predict_regression, confusion_matrix, scatterplot,
    summarize, metatable, heatmap)
from .visuals import _custom_palettes
from ._format import (SampleEstimatorDirFmt,
                      BooleanSeriesFormat,
                      BooleanSeriesDirectoryFormat,
                      ImportanceFormat,
                      ImportanceDirectoryFormat,
                      PredictionsFormat,
                      PredictionsDirectoryFormat,
                      ProbabilitiesFormat,
                      ProbabilitiesDirectoryFormat,
                      TrueTargetsDirectoryFormat)

from ._type import (ClassifierPredictions, RegressorPredictions,
                    SampleEstimator, BooleanSeries, Importance,
                    Classifier, Regressor, Probabilities,
                    TrueTargets)
import q2_sample_classifier
from q2_sample_classifier.classify import shapely_values

citations = Citations.load('citations.bib', package='q2_sample_classifier')

plugin = Plugin(
    name='sample-classifier',
    version=q2_sample_classifier.__version__,
    website="https://github.com/qiime2/q2-sample-classifier",
    package='q2_sample_classifier',
    description=(
        'This QIIME 2 plugin supports methods for supervised classification '
        'and regression of sample metadata, and other supervised machine '
        'learning methods.'),
    short_description=(
        'Plugin for machine learning prediction of sample metadata.'),
    citations=[citations['Bokulich306167'], citations['pedregosa2011scikit']]
)

description = ('Predicts a {0} sample metadata column using a {1}. Splits '
               'input data into training and test sets. The training set is '
               'used to train and test the estimator using a stratified '
               'k-fold cross-validation scheme. This includes optional steps '
               'for automated feature extraction and hyperparameter '
               'optimization. The test set validates classification accuracy '
               'of the optimized estimator. Outputs classification results '
               'for test set. For more details on the learning algorithm, '
               'see http://scikit-learn.org/stable/supervised_learning.html')

ncv_description = ('Predicts a {0} sample metadata column using a {1}. Uses '
                   'nested stratified k-fold cross validation for automated '
                   'hyperparameter optimization and sample prediction. '
                   'Outputs predicted values for each input sample, and '
                   'relative importance of each feature for model accuracy.')

cv_description = ('Fit a supervised learning {0}. Outputs the fit estimator '
                  '(for prediction of test samples and/or unknown samples) '
                  'and the relative importance of each feature for model '
                  'accuracy. Optionally use k-fold cross-validation for '
                  'automatic recursive feature elimination and hyperparameter '
                  'tuning.')

predict_description = (
    'Use trained estimator to predict target values for new samples. '
    'These will typically be unseen samples, e.g., test data (derived '
    'manually or from split_table) or samples with unknown values, but '
    'can theoretically be any samples present in a feature table that '
    'contain overlapping features with the feature table used to train '
    'the estimator.')

inputs = {'table': FeatureTable[Frequency]}

input_descriptions = {'table': 'Feature table containing all features that '
                               'should be used for target prediction.',
                      'probabilities': 'Predicted class probabilities for '
                                       'each input sample.'}

parameters = {
    'base': {
        'random_state': Int,
        'n_jobs': Int,
        'n_estimators': Int % Range(1, None),
        'missing_samples': Str % Choices(['error', 'ignore'])},
    'splitter': {
        'test_size': Float % Range(0.0, 1.0, inclusive_end=False,
                                   inclusive_start=True)},
    'rfe': {
        'step': Float % Range(0.0, 1.0, inclusive_end=False,
                              inclusive_start=False),
        'optimize_feature_selection': Bool},
    'cv': {
        'cv': Int % Range(1, None),
        'parameter_tuning': Bool},
    'modified_metadata': {
        'metadata': Metadata,
        'column': Str},
    'regressor': {'stratify': Bool}
}

parameter_descriptions = {
    'base': {'random_state': 'Seed used by random number generator.',
             'n_jobs': 'Number of jobs to run in parallel.',
             'n_estimators': (
                'Number of trees to grow for estimation. More trees will '
                'improve predictive accuracy up to a threshold level, '
                'but will also increase time and memory requirements. This '
                'parameter only affects ensemble estimators, such as Random '
                'Forest, AdaBoost, ExtraTrees, and GradientBoosting.'),
             'missing_samples': (
                'How to handle missing samples in metadata. "error" will fail '
                'if missing samples are detected. "ignore" will cause the '
                'feature table and metadata to be filtered, so that only '
                'samples found in both files are retained.')},
    'splitter': {
        'test_size': ('Fraction of input samples to exclude from training set '
                      'and use for classifier testing.')},
    'rfe': {
        'step': ('If optimize_feature_selection is True, step is the '
                 'percentage of features to remove at each iteration.'),
        'optimize_feature_selection': ('Automatically optimize input feature '
                                       'selection using recursive feature '
                                       'elimination.')},
    'cv': {
        'cv': 'Number of k-fold cross-validations to perform.',
        'parameter_tuning': ('Automatically tune hyperparameters using random '
                             'grid search.')},
    'regressor': {
        'stratify': ('Evenly stratify training and test data among metadata '
                     'categories. If True, all values in column must match '
                     'at least two samples.')},
    'estimator': {
        'estimator': 'Estimator method to use for sample prediction.'}
}

classifiers = Str % Choices(
    ['RandomForestClassifier', 'ExtraTreesClassifier',
     'GradientBoostingClassifier', 'AdaBoostClassifier',
     'KNeighborsClassifier', 'LinearSVC', 'SVC'])

regressors = Str % Choices(
    ['RandomForestRegressor', 'ExtraTreesRegressor',
     'GradientBoostingRegressor', 'AdaBoostRegressor', 'ElasticNet',
     'Ridge', 'Lasso', 'KNeighborsRegressor', 'LinearSVR', 'SVR'])

output_descriptions = {
    'predictions': 'Predicted target values for each input sample.',
    'feature_importance': 'Importance of each input feature to model accuracy.'
}

pipeline_parameters = {
    **parameters['base'],
    **parameters['rfe'],
    **parameters['splitter'],
    **parameters['cv']}

classifier_pipeline_parameters = {
    **pipeline_parameters,
    'metadata': MetadataColumn[Categorical],
    'estimator': classifiers,
    'palette': Str % Choices(_custom_palettes().keys())}

regressor_pipeline_parameters = {
    **pipeline_parameters,
    'metadata': MetadataColumn[Numeric],
    **parameters['regressor'],
    'estimator': regressors}

pipeline_parameter_descriptions = {
    **parameter_descriptions['base'],
    **parameter_descriptions['rfe'],
    **parameter_descriptions['splitter'],
    **parameter_descriptions['estimator'],
    **parameter_descriptions['cv']}

classifier_pipeline_parameter_descriptions = {
    **pipeline_parameter_descriptions,
    'metadata': 'Categorical metadata column to use as prediction target.',
    'palette': 'The color palette to use for plotting.'}

regressor_pipeline_parameter_descriptions = {
    **pipeline_parameter_descriptions,
    **parameter_descriptions['regressor'],
    'metadata': 'Numeric metadata column to use as prediction target.'}

pipeline_outputs = [
    ('model_summary', Visualization),
    ('accuracy_results', Visualization)]

regressor_pipeline_outputs = [
    ('sample_estimator', SampleEstimator[Regressor]),
    ('feature_importance', FeatureData[Importance]),
    ('predictions', SampleData[RegressorPredictions])] + pipeline_outputs

pipeline_output_descriptions = {
    'sample_estimator': 'Trained sample estimator.',
    **output_descriptions,
    'model_summary': 'Summarized parameter and (if enabled) feature '
                     'selection information for the trained estimator.',
    'accuracy_results': 'Accuracy results visualization.'}


plugin.pipelines.register_function(
    function=classify_samples,
    inputs=inputs,
    parameters=classifier_pipeline_parameters,
    outputs=[('sample_estimator', SampleEstimator[Classifier]),
             ('feature_importance', FeatureData[Importance]),
             ('predictions', SampleData[ClassifierPredictions])
             ] + pipeline_outputs + [
        ('probabilities', SampleData[Probabilities]),
        ('heatmap', Visualization),
        ('training_targets', SampleData[TrueTargets]),
        ('test_targets', SampleData[TrueTargets])],
    input_descriptions={'table': input_descriptions['table']},
    parameter_descriptions=classifier_pipeline_parameter_descriptions,
    output_descriptions={
        **pipeline_output_descriptions,
        'probabilities': input_descriptions['probabilities'],
        'heatmap': 'A heatmap of the top 50 most important features from the '
                   'table.',
        'training_targets': 'Series containing true target values of '
        'train samples',
        'test_targets': 'Series containing true target values '
        'of test samples'},
    name='Train and test a cross-validated supervised learning classifier.',
    description=description.format(
        'categorical', 'supervised learning classifier')
)


plugin.pipelines.register_function(
    function=classify_samples_from_dist,
    inputs={'distance_matrix': DistanceMatrix},
    parameters={
        'metadata': MetadataColumn[Categorical],
        'k': Int,
        'cv': parameters['cv']['cv'],
        'random_state': parameters['base']['random_state'],
        'n_jobs': parameters['base']['n_jobs'],
        'palette': Str % Choices(_custom_palettes().keys()),
    },
    outputs=[
        ('predictions', SampleData[ClassifierPredictions]),
        ('accuracy_results', Visualization),
    ],
    input_descriptions={'distance_matrix': 'a distance matrix'},
    parameter_descriptions={
        'metadata': 'Categorical metadata column to use as prediction target.',
        'k': 'Number of nearest neighbors',
        'cv': parameter_descriptions['cv']['cv'],
        'random_state': parameter_descriptions['base']['random_state'],
        'n_jobs': parameter_descriptions['base']['n_jobs'],
        'palette': 'The color palette to use for plotting.',
    },
    output_descriptions={
        'predictions': 'leave one out predictions for each sample',
        'accuracy_results': 'Accuracy results visualization.',
    },
    name=('Run k-nearest-neighbors on a labeled distance matrix.'),
    description=(
        'Run k-nearest-neighbors on a labeled distance matrix.'
        ' Return cross-validated (leave one out) predictions and '
        ' accuracy. k = 1 by default'
    )
)


plugin.pipelines.register_function(
    function=regress_samples,
    inputs=inputs,
    parameters=regressor_pipeline_parameters,
    outputs=regressor_pipeline_outputs,
    input_descriptions={'table': input_descriptions['table']},
    parameter_descriptions=regressor_pipeline_parameter_descriptions,
    output_descriptions=pipeline_output_descriptions,
    name='Train and test a cross-validated supervised learning regressor.',
    description=description.format(
        'continuous', 'supervised learning regressor')
)


plugin.methods.register_function(
    function=regress_samples_ncv,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['cv'],
        'metadata': MetadataColumn[Numeric],
        **parameters['regressor'],
        'estimator': regressors},
    outputs=[('predictions', SampleData[RegressorPredictions]),
             ('feature_importance', FeatureData[Importance])],
    input_descriptions={'table': input_descriptions['table']},
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['cv'],
        **parameter_descriptions['regressor'],
        'metadata': 'Numeric metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
    output_descriptions=output_descriptions,
    name='Nested cross-validated supervised learning regressor.',
    description=ncv_description.format(
        'continuous', 'supervised learning regressor')
)

plugin.methods.register_function(
    function=classify_samples_ncv,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['cv'],
        'metadata': MetadataColumn[Categorical],
        'estimator': classifiers},
    outputs=[('predictions', SampleData[ClassifierPredictions]),
             ('feature_importance', FeatureData[Importance]),
             ('probabilities', SampleData[Probabilities])],
    input_descriptions={'table': input_descriptions['table']},
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['cv'],
        'metadata': 'Categorical metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
    output_descriptions={**output_descriptions,
                         'probabilities': input_descriptions['probabilities']},
    name='Nested cross-validated supervised learning classifier.',
    description=ncv_description.format(
        'categorical', 'supervised learning classifier')
)


plugin.methods.register_function(
    function=fit_classifier,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['rfe'],
        **parameters['cv'],
        'metadata': MetadataColumn[Categorical],
        'estimator': classifiers},
    outputs=[('sample_estimator', SampleEstimator[Classifier]),
             ('feature_importance', FeatureData[Importance])],
    input_descriptions={'table': input_descriptions['table']},
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['rfe'],
        **parameter_descriptions['cv'],
        'metadata': 'Numeric metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
    output_descriptions={
        'feature_importance': output_descriptions['feature_importance'],
        'sample_estimator': 'Trained sample classifier.'},
    name='Fit a supervised learning classifier.',
    description=cv_description.format('classifier')
)


plugin.methods.register_function(
    function=fit_regressor,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['rfe'],
        **parameters['cv'],
        'metadata': MetadataColumn[Numeric],
        'estimator': regressors},
    outputs=[('sample_estimator', SampleEstimator[Regressor]),
             ('feature_importance', FeatureData[Importance])],
    input_descriptions={'table': input_descriptions['table']},
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['rfe'],
        **parameter_descriptions['cv'],
        'metadata': 'Numeric metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
    output_descriptions={
        'feature_importance': output_descriptions['feature_importance']},
    name='Fit a supervised learning regressor.',
    description=cv_description.format('regressor')
)


plugin.methods.register_function(
    function=predict_classification,
    inputs={**inputs, 'sample_estimator': SampleEstimator[Classifier]},
    parameters={'n_jobs': parameters['base']['n_jobs']},
    outputs=[('predictions', SampleData[ClassifierPredictions]),
             ('probabilities', SampleData[Probabilities])],
    input_descriptions={
        'table': input_descriptions['table'],
        'sample_estimator': 'Sample classifier trained with fit_classifier.'},
    parameter_descriptions={
        'n_jobs': parameter_descriptions['base']['n_jobs']},
    output_descriptions={
        'predictions': 'Predicted target values for each input sample.',
        'probabilities': input_descriptions['probabilities']},
    name='Use trained classifier to predict target values for new samples.',
    description=predict_description
)


plugin.methods.register_function(
    function=predict_regression,
    inputs={**inputs, 'sample_estimator': SampleEstimator[Regressor]},
    parameters={'n_jobs': parameters['base']['n_jobs']},
    outputs=[('predictions', SampleData[RegressorPredictions])],
    input_descriptions={
        'table': input_descriptions['table'],
        'sample_estimator': 'Sample regressor trained with fit_regressor.'},
    parameter_descriptions={
        'n_jobs': parameter_descriptions['base']['n_jobs']},
    output_descriptions={
        'predictions': 'Predicted target values for each input sample.'},
    name='Use trained regressor to predict target values for new samples.',
    description=predict_description
)


plugin.visualizers.register_function(
    function=scatterplot,
    inputs={'predictions': SampleData[RegressorPredictions]},
    parameters={
        'truth': MetadataColumn[Numeric],
        'missing_samples': parameters['base']['missing_samples']},
    input_descriptions={'predictions': (
        'Predicted values to plot on y axis. Must be predictions of '
        'numeric data produced by a sample regressor.')},
    parameter_descriptions={
        'truth': 'Metadata column (true values) to plot on x axis.',
        'missing_samples': parameter_descriptions['base']['missing_samples']},
    name='Make 2D scatterplot and linear regression of regressor predictions.',
    description='Make a 2D scatterplot and linear regression of predicted vs. '
                'true values for a set of samples predicted using a sample '
                'regressor.'
)


plugin.visualizers.register_function(
    function=confusion_matrix,
    inputs={'predictions': SampleData[ClassifierPredictions],
            'probabilities': SampleData[Probabilities]},
    parameters={
        'truth': MetadataColumn[Categorical],
        'missing_samples': parameters['base']['missing_samples'],
        'vmin': Float | Str % Choices(['auto']),
        'vmax': Float | Str % Choices(['auto']),
        'palette': Str % Choices(_custom_palettes().keys())},
    input_descriptions={
        'predictions': 'Predicted values to plot on x axis. Should be '
                       'predictions of categorical data produced by a sample '
                       'classifier.',
        'probabilities': input_descriptions['probabilities']},
    parameter_descriptions={
        'truth': 'Metadata column (true values) to plot on y axis.',
        'missing_samples': parameter_descriptions['base']['missing_samples'],
        'vmin': 'The minimum value to use for anchoring the colormap. If '
        '"auto", vmin is set to the minimum value in the data.',
        'vmax': 'The maximum value to use for anchoring the colormap. If '
        '"auto", vmax is set to the maximum value in the data.',
        'palette': 'The color palette to use for plotting.'},
    name='Make a confusion matrix from sample classifier predictions.',
    description='Make a confusion matrix and calculate accuracy of predicted '
                'vs. true values for a set of samples classified using a '
                'sample classifier. If per-sample class probabilities are '
                'provided, will also generate Receiver Operating '
                'Characteristic curves and calculate area under the curve for '
                'each class.'
)


T = TypeMatch([Frequency, RelativeFrequency, PresenceAbsence, Balance,
               PercentileNormalized, Design])
plugin.methods.register_function(
    function=split_table,
    inputs={'table': FeatureTable[T]},
    parameters={
        'random_state': parameters['base']['random_state'],
        'missing_samples': parameters['base']['missing_samples'],
        **parameters['splitter'],
        'metadata': MetadataColumn[Numeric | Categorical],
        **parameters['regressor']},
    outputs=[('training_table', FeatureTable[T]),
             ('test_table', FeatureTable[T]),
             ('training_targets', SampleData[TrueTargets]),
             ('test_targets', SampleData[TrueTargets])],
    input_descriptions={'table': 'Feature table containing all features that '
                        'should be used for target prediction.'},
    parameter_descriptions={
        'random_state': parameter_descriptions['base']['random_state'],
        'missing_samples': parameter_descriptions['base']['missing_samples'],
        **parameter_descriptions['splitter'],
        **parameter_descriptions['regressor'],
        'metadata': 'Numeric metadata column to use as prediction target.'},
    output_descriptions={
        'training_table': 'Feature table containing training samples',
        'test_table': 'Feature table containing test samples',
        'training_targets': 'Series containing true target values of '
        'train samples',
        'test_targets': 'Series containing true target values of '
        'test samples'},
    name='Split a feature table into training and testing sets.',
    description=(
        'Split a feature table into training and testing sets. By default '
        'stratifies training and test sets on a metadata column, such that '
        'values in that column are evenly represented across training and '
        'test sets.')
)


plugin.visualizers.register_function(
    function=summarize,
    inputs={'sample_estimator': SampleEstimator[Classifier | Regressor]},
    parameters={},
    input_descriptions={
        'sample_estimator': 'Sample estimator trained with fit_classifier or '
                            'fit_regressor.'},
    parameter_descriptions={},
    name='Summarize parameter and feature extraction information for a '
         'trained estimator.',
    description='Summarize parameter and feature extraction information for a '
                'trained estimator.'
)


plugin.pipelines.register_function(
    function=metatable,
    inputs=inputs,
    parameters={'metadata': Metadata,
                'missing_samples': parameters['base']['missing_samples'],
                'missing_values': Str % Choices(
                    ['drop_samples', 'drop_features', 'error', 'fill']),
                'drop_all_unique': Bool},
    outputs=[('converted_table', FeatureTable[Frequency])],
    input_descriptions={'table': input_descriptions['table']},
    parameter_descriptions={
        'metadata': 'Metadata file to convert to feature table.',
        'missing_samples': parameter_descriptions['base']['missing_samples'],
        'missing_values': (
            'How to handle missing values (nans) in metadata. Either '
            '"drop_samples" with missing values, "drop_features" with missing '
            'values, "fill" missing values with zeros, or "error" if '
            'any missing values are found.'),
        'drop_all_unique': 'If True, columns that contain a unique value for '
                           'every ID will be dropped.'
    },
    output_descriptions={'converted_table': 'Converted feature table'},
    name='Convert (and merge) positive numeric metadata (in)to feature table.',
    description='Convert numeric sample metadata from TSV file into a feature '
                'table. Optionally merge with an existing feature table. Only '
                'numeric metadata will be converted; categorical columns will '
                'be silently dropped. By default, if a table is used as input '
                'only samples found in both the table and metadata '
                '(intersection) are merged, and others are silently dropped. '
                'Set missing_samples="error" to raise an error if samples '
                'found in the table are missing from the metadata file. The '
                'metadata file can always contain a superset of samples. Note '
                'that columns will be dropped if they are non-numeric, '
                'contain no unique values (zero '
                'variance), contain only empty cells, or contain negative '
                'values. This method currently only converts '
                'postive numeric metadata into feature data. Tip: convert '
                'categorical columns to dummy variables to include them in '
                'the output feature table.'
)


plugin.pipelines.register_function(
    function=heatmap,
    inputs={**inputs, 'importance': FeatureData[Importance]},
    parameters={'sample_metadata': MetadataColumn[Categorical],
                'feature_metadata': MetadataColumn[Categorical],
                'feature_count': Int % Range(0, None),
                'importance_threshold': Float % Range(0, None),
                'group_samples': Bool,
                'normalize': Bool,
                'missing_samples': parameters['base']['missing_samples'],
                'metric': Str % Choices(heatmap_choices['metric']),
                'method': Str % Choices(heatmap_choices['method']),
                'cluster': Str % Choices(heatmap_choices['cluster']),
                'color_scheme': Str % Choices(heatmap_choices['color_scheme']),
                },
    outputs=[('heatmap', Visualization),
             ('filtered_table', FeatureTable[Frequency])],
    input_descriptions={'table': input_descriptions['table'],
                        'importance': 'Feature importances.'},
    parameter_descriptions={
        'sample_metadata': 'Sample metadata column to use for sample labeling '
                           'or grouping.',
        'feature_metadata': 'Feature metadata (e.g., taxonomy) to use for '
                            'labeling features in the heatmap.',
        'feature_count': 'Filter feature table to include top N most '
                         'important features. Set to zero to include all '
                         'features.',
        'importance_threshold': 'Filter feature table to exclude any features '
                                'with an importance score less than this '
                                'threshold. Set to zero to include all '
                                'features.',
        'group_samples': 'Group samples by sample metadata.',
        'normalize': 'Normalize the feature table by adding a psuedocount '
                     'of 1 and then taking the log10 of the table.',
        'missing_samples': parameter_descriptions['base']['missing_samples'],
        'metric': 'Metrics exposed by seaborn (see http://seaborn.pydata.org/'
                  'generated/seaborn.clustermap.html#seaborn.clustermap for '
                  'more detail).',
        'method': 'Clustering methods exposed by seaborn (see http://seaborn.'
                  'pydata.org/generated/seaborn.clustermap.html#seaborn.clust'
                  'ermap for more detail).',
        'cluster': 'Specify which axes to cluster.',
        'color_scheme': 'Color scheme for heatmap.',
    },
    output_descriptions={
        'heatmap': 'Heatmap of important features.',
        'filtered_table': 'Filtered feature table containing data displayed '
                          'in heatmap.'},
    name='Generate heatmap of important features.',
    description='Generate a heatmap of important features. Features are '
                'filtered based on importance scores; samples are optionally '
                'grouped by sample metadata; and a heatmap is generated that '
                'displays (normalized) feature abundances per sample.'
)

plugin.methods.register_function(
    function=shapely_values,
    inputs={**inputs, 'sample_estimator': SampleEstimator[Classifier]},
    parameters={},
    outputs=[('shap', SampleData[Probabilities])],
    input_descriptions={
        'table': input_descriptions['table'],
        'sample_estimator': 'Sample classifier trained with fit_classifier.'},
    output_descriptions={
        'shap': 'Contributions of each feature towards the prediction.'},
    name='Use trained classifier to compute Shapely values new samples.',
    description=(
        "Computes shapely values, which measures the contribution of each feature "
        "for a given sample label prediction."
    )
)

# Registrations
plugin.register_semantic_types(
    SampleEstimator, BooleanSeries, Importance, ClassifierPredictions,
    RegressorPredictions, Classifier, Regressor, Probabilities, TrueTargets)
plugin.register_semantic_type_to_format(
    SampleEstimator[Classifier],
    artifact_format=SampleEstimatorDirFmt)
plugin.register_semantic_type_to_format(
    SampleEstimator[Regressor],
    artifact_format=SampleEstimatorDirFmt)
plugin.register_semantic_type_to_format(
    SampleData[BooleanSeries],
    artifact_format=BooleanSeriesDirectoryFormat)
plugin.register_semantic_type_to_format(
    SampleData[RegressorPredictions],
    artifact_format=PredictionsDirectoryFormat)
plugin.register_semantic_type_to_format(
    SampleData[ClassifierPredictions],
    artifact_format=PredictionsDirectoryFormat)
plugin.register_semantic_type_to_format(
    FeatureData[Importance],
    artifact_format=ImportanceDirectoryFormat)
plugin.register_semantic_type_to_format(
    SampleData[Probabilities],
    artifact_format=ProbabilitiesDirectoryFormat)
plugin.register_semantic_type_to_format(
    SampleData[TrueTargets],
    artifact_format=TrueTargetsDirectoryFormat)
plugin.register_formats(
    SampleEstimatorDirFmt, BooleanSeriesFormat, BooleanSeriesDirectoryFormat,
    ImportanceFormat, ImportanceDirectoryFormat, PredictionsFormat,
    PredictionsDirectoryFormat, ProbabilitiesFormat,
    ProbabilitiesDirectoryFormat,
    TrueTargetsDirectoryFormat)
importlib.import_module('q2_sample_classifier._transformer')
