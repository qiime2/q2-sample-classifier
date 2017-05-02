#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Int, Str, Float, Range, Bool, Plugin, Metadata
from q2_types.feature_table import FeatureTable, Frequency
from .classify import (classify_random_forest, regress_random_forest)
import q2_sample_classifier


plugin = Plugin(
    name='sample-classifier',
    version=q2_sample_classifier.__version__,
    website="https://github.com/nbokulich/q2-sample-classifier",
    package='q2_sample_classifier'
)


description = ('Predict {0} sample metadata classes using a {1}. Splits input '
               'data into training and test  sets. The training set is used '
               'to train and test the  classifier using a stratified k-fold '
               'cross-validation scheme. This includes optional steps for '
               'automated feature extraction and hyperparameter optimization. '
               'The test set validates classification accuracy of the '
               'optimized estimator. Outputs classification results for test '
               'set, and optionally a trained estimator to use on additional '
               'unknown samples.')

inputs = {'table': FeatureTable[Frequency]}

parameters = {'metadata': Metadata,
              'category': Str,
              'test_size': Float % Range(0.0, 1.0, inclusive_end=False,
                                         inclusive_start=False),
              'step': Float % Range(0.0, 1.0, inclusive_end=False,
                                    inclusive_start=False),
              'cv': Int % Range(1, None),
              'random_state': Int,
              'n_jobs': Int,
              'parameter_tuning': Bool}

input_descriptions = {'table': ('Feature table containing all features that '
                                'should be used for target prediction.')}

parameter_descriptions = {
    'metadata': 'Sample metadata to use as prediction targets.',
    'category': 'Metadata category to use for training and prediction.',
    'test_size': ('Fraction of input samples to exclude from training set '
                  'and use for classifier testing.'),
    'step': ('If optimize_feature_selection is True, step is the '
             'percentage of features to remove at each iteration.'),
    'cv': 'Number of k-fold cross-validations to perform.',
    'random_state': 'Seed used by random number generator.',
    'n_jobs': 'Number of jobs to run in parallel.',
    'n_estimators': ('Number of random forests to grow for estimation. '
                     'More trees will improve predictive accuracy up to '
                     'a threshold level, but will also increase time and '
                     'memory requirements.'),
    'optimize_feature_selection': ('Automatically optimize input feature '
                                   'selection using recursive feature '
                                   'elimination?'),
    'parameter_tuning': ('Automatically tune hyperparameters using random '
                         'grid search?')
}


random_forest_parameters = {
    'n_estimators': Int % Range(1, None), 'optimize_feature_selection': Bool
}


random_forest_parameter_descriptions = {
    'n_estimators': ('Number of random forests to grow for estimation. '
                     'More trees will improve predictive accuracy up to '
                     'a threshold level, but will also increase time and '
                     'memory requirements.'),
    'optimize_feature_selection': ('Automatically optimize input feature '
                                   'selection using recursive feature '
                                   'elimination?')
}


plugin.visualizers.register_function(
    function=classify_random_forest,
    inputs=inputs,
    parameters={**parameters, **random_forest_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **random_forest_parameter_descriptions},
    name='Random forest sample classifier',
    description=description.format('categorical', 'random forest classifier')
)


plugin.visualizers.register_function(
    function=regress_random_forest,
    inputs=inputs,
    parameters={**parameters, **random_forest_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **random_forest_parameter_descriptions},
    name='Random forest regressor',
    description=description.format('continuous', 'random forest regressor')
)
