# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import importlib

from qiime2.plugin import (
    Int, Str, Float, Range, Bool, Plugin, Metadata, Choices, MetadataColumn,
    Numeric, Categorical, Citations)
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData
from .classify import (
    classify_samples, regress_samples, maturity_index, regress_samples_ncv,
    classify_samples_ncv, fit_classifier, fit_regressor, split_table, predict,
    confusion_matrix, scatterplot, summarize)
from .visuals import _custom_palettes
from ._format import (SampleEstimatorDirFmt,
                      BooleanSeriesFormat,
                      BooleanSeriesDirectoryFormat,
                      ImportanceFormat,
                      ImportanceDirectoryFormat,
                      PredictionsFormat,
                      PredictionsDirectoryFormat)

from ._type import Predictions, SampleEstimator, BooleanSeries, Importance
import q2_sample_classifier

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
    citations=[citations['Bokulich306167']]
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

inputs = {'table': FeatureTable[Frequency]}

input_descriptions = {'table': ('Feature table containing all features that '
                                'should be used for target prediction.')}

sample_estimator = {'sample_estimator': SampleEstimator}

sample_estimator_description = {
    'sample_estimator': 'Sample estimator trained with fit_classifier or '
                        'fit_regressor.'}

parameters = {
    'base': {
        'random_state': Int,
        'n_jobs': Int,
        'n_estimators': Int % Range(1, None),
        'missing_samples': Str % Choices(['error', 'ignore'])},
    'splitter': {
        'test_size': Float % Range(0.0, 1.0, inclusive_end=False,
                                   inclusive_start=False)},
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

outputs = [('predictions', SampleData[Predictions]),
           ('feature_importance', FeatureData[Importance])]

output_descriptions = {
    'predictions': 'Predicted target values for each input sample.',
    'feature_importance': 'Importance of each input feature to model accuracy.'
}


plugin.visualizers.register_function(
    function=classify_samples,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['rfe'],
        **parameters['splitter'],
        **parameters['cv'],
        'metadata': MetadataColumn[Categorical],
        'estimator': classifiers,
        'palette': Str % Choices(_custom_palettes().keys())},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['rfe'],
        **parameter_descriptions['splitter'],
        **parameter_descriptions['cv'],
        'metadata': ('Categorical metadata column to use as prediction '
                     'target.'),
        **parameter_descriptions['estimator'],
        'palette': 'The color palette to use for plotting.'},
    name='Train and test a cross-validated supervised learning classifier.',
    description=description.format(
        'categorical', 'supervised learning classifier')
)


plugin.visualizers.register_function(
    function=regress_samples,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['rfe'],
        **parameters['splitter'],
        **parameters['cv'],
        'metadata': MetadataColumn[Numeric],
        **parameters['regressor'],
        'estimator': regressors},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['rfe'],
        **parameter_descriptions['splitter'],
        **parameter_descriptions['cv'],
        **parameter_descriptions['regressor'],
        'metadata': 'Numeric metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
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
    outputs=outputs,
    input_descriptions=input_descriptions,
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
    outputs=outputs,
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['cv'],
        'metadata': 'Categorical metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
    output_descriptions=output_descriptions,
    name='Nested cross-validated supervised learning classifier.',
    description=ncv_description.format(
        'categorical', 'supervised learning classifier')
)


fitter_outputs = [('sample_estimator', SampleEstimator),
                  ('feature_importance', FeatureData[Importance])]

plugin.methods.register_function(
    function=fit_classifier,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['rfe'],
        **parameters['cv'],
        'metadata': MetadataColumn[Categorical],
        'estimator': classifiers},
    outputs=fitter_outputs,
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['rfe'],
        **parameter_descriptions['cv'],
        'metadata': 'Numeric metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
    output_descriptions={
        'feature_importance': output_descriptions['feature_importance'],
        **sample_estimator_description},
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
    outputs=fitter_outputs,
    input_descriptions=input_descriptions,
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
    function=predict,
    inputs={**inputs, **sample_estimator},
    parameters={
        'n_jobs': parameters['base']['n_jobs']},
    outputs=[('predictions', SampleData[Predictions])],
    input_descriptions={**input_descriptions, **sample_estimator_description},
    parameter_descriptions={
        'n_jobs': parameter_descriptions['base']['n_jobs']},
    output_descriptions={
        'predictions': 'Predicted target values for each input sample.'},
    name='Use trained estimator to predict target values for new samples.',
    description=(
        'Use trained estimator to predict target values for new samples. '
        'These will typically be unseen samples, e.g., test data (derived '
        'manually or from split_table) or samples with unknown values, but '
        'can theoretically be any samples present in a feature table that '
        'contain overlapping features with the feature table used to train '
        'the estimator.')
)


plugin.visualizers.register_function(
    function=scatterplot,
    inputs={'predictions': SampleData[Predictions]},
    parameters={
        'truth': MetadataColumn[Numeric],
        'missing_samples': parameters['base']['missing_samples']},
    input_descriptions={'predictions': 'Predicted values to plot on y axis'},
    parameter_descriptions={
        'truth': 'Metadata column (true values) to plot on x axis.',
        'missing_samples': parameter_descriptions['base']['missing_samples']},
    name='Make a 2D scatterplot and linear regression.',
    description='Make a 2D scatterplot and linear regression of predicted vs. '
                'true values for a set of samples.'
)


plugin.visualizers.register_function(
    function=confusion_matrix,
    inputs={'predictions': SampleData[Predictions]},
    parameters={
        'truth': MetadataColumn[Categorical],
        'missing_samples': parameters['base']['missing_samples'],
        'palette': Str % Choices(_custom_palettes().keys())},
    input_descriptions={'predictions': 'Predicted values to plot on x axis'},
    parameter_descriptions={
        'truth': 'Metadata column (true values) to plot on y axis.',
        'missing_samples': parameter_descriptions['base']['missing_samples'],
        'palette': 'The color palette to use for plotting.'},
    name='Make a confusion matrix.',
    description='Make a confusion matrix and calculate accuracy of predicted '
                'vs. true values for a set of samples.'
)


plugin.methods.register_function(
    function=split_table,
    inputs=inputs,
    parameters={
        'random_state': parameters['base']['random_state'],
        'missing_samples': parameters['base']['missing_samples'],
        **parameters['splitter'],
        'metadata': MetadataColumn[Numeric | Categorical],
        **parameters['regressor']},
    outputs=[('training_table', FeatureTable[Frequency]),
             ('test_table', FeatureTable[Frequency])],
    input_descriptions=input_descriptions,
    parameter_descriptions={
        'random_state': parameter_descriptions['base']['random_state'],
        'missing_samples': parameter_descriptions['base']['missing_samples'],
        **parameter_descriptions['splitter'],
        **parameter_descriptions['regressor'],
        'metadata': 'Numeric metadata column to use as prediction target.'},
    output_descriptions={
        'training_table': 'Feature table containing training samples',
        'test_table': 'Feature table containing test samples'},
    name='Split a feature table into training and testing sets.',
    description=(
        'Split a feature table into training and testing sets. By default '
        'stratifies training and test sets on a metadata column, such that '
        'values in that column are evenly represented across training and '
        'test sets.')
)


plugin.visualizers.register_function(
    function=summarize,
    inputs=sample_estimator,
    parameters={},
    input_descriptions=sample_estimator_description,
    parameter_descriptions={},
    name='Summarize parameter and feature extraction information for a '
         'trained estimator.',
    description='Summarize parameter and feature extraction information for a '
                'trained estimator.'
)


plugin.visualizers.register_function(
    function=maturity_index,
    inputs=inputs,
    parameters={'group_by': Str,
                'control': Str,
                'estimator': regressors,
                **parameters['base'],
                **parameters['rfe'],
                **parameters['cv'],
                **parameters['splitter'],
                'metadata': Metadata,
                'column': Str,
                **parameters['regressor'],
                'maz_stats': Bool,
                },
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['rfe'],
        **parameter_descriptions['cv'],
        **parameter_descriptions['splitter'],
        'column': 'Numeric metadata column to use as prediction target.',
        'group_by': ('Categorical metadata column to use for plotting and '
                     'significance testing between main treatment groups.'),
        'control': (
            'Value of group_by to use as control group. The regression model '
            'will be trained using only control group data, and the maturity '
            'scores of other groups consequently will be assessed relative to '
            'this group.'),
        'estimator': 'Regression model to use for prediction.',
        **parameter_descriptions['regressor'],
        'maz_stats': 'Calculate anova and pairwise tests on MAZ scores.',
    },
    name='Microbial maturity index prediction.',
    description=('Calculates a "microbial maturity" index from a regression '
                 'model trained on feature data to predict a given continuous '
                 'metadata column, e.g., to predict age as a function of '
                 'microbiota composition. The model is trained on a subset of '
                 'control group samples, then predicts the column value for '
                 'all samples. This visualization computes maturity index '
                 'z-scores to compare relative "maturity" between each group, '
                 'as described in doi:10.1038/nature13421. This method can '
                 'be used to predict between-group differences in relative '
                 'trajectory across any type of continuous metadata gradient, '
                 'e.g., intestinal microbiome development by age, microbial '
                 'succession during wine fermentation, or microbial community '
                 'differences along environmental gradients, as a function of '
                 'two or more different "treatment" groups.'),
    citations=[citations['subramanian2014persistent']]
)


# Registrations
plugin.register_semantic_types(
    SampleEstimator, BooleanSeries, Importance, Predictions)
plugin.register_semantic_type_to_format(
    SampleEstimator,
    artifact_format=SampleEstimatorDirFmt)
plugin.register_semantic_type_to_format(
    SampleData[BooleanSeries],
    artifact_format=BooleanSeriesDirectoryFormat)
plugin.register_semantic_type_to_format(
    SampleData[Predictions],
    artifact_format=PredictionsDirectoryFormat)
plugin.register_semantic_type_to_format(
    FeatureData[Importance],
    artifact_format=ImportanceDirectoryFormat)
plugin.register_formats(
    SampleEstimatorDirFmt, BooleanSeriesFormat, BooleanSeriesDirectoryFormat,
    ImportanceFormat, ImportanceDirectoryFormat, PredictionsFormat,
    PredictionsDirectoryFormat)
importlib.import_module('q2_sample_classifier._transformer')
