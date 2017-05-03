#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import (Int, Str, Float, Range, Bool, Plugin, Metadata,
                           Choices)
from q2_types.feature_table import FeatureTable, Frequency
from .classify import (
    classify_random_forest, regress_random_forest, classify_linearSVC,
    regress_linearSVR, regress_SVR, classify_SVC, classify_kneighbors,
    regress_ridge, )#regress_lasso, regress_elasticnet)
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


svm_parameters = {
    'kernel': Str % Choices(['linear', 'poly', 'rbf', 'sigmoid'])
}


svm_parameter_descriptions = {
    'kernel': 'Specifies the kernel type to be used in the algorithm.'
}


neighbors_parameters = {
    'algorithm': Str % Choices(['ball_tree', 'kd_tree', 'brute', 'auto'])
}


neighbors_parameter_descriptions = {
    'algorithm': ('Algorithm used to compute the nearest neighbors. Default, '
    'auto, will attempt to decide the most appropriate algorithm based on the '
    'values passed to fit method.')
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


plugin.visualizers.register_function(
    function=classify_linearSVC,
    inputs=inputs,
    parameters=parameters,
    input_descriptions=input_descriptions,
    parameter_descriptions=parameter_descriptions,
    name='Linear support vector machine classifier',
    description=description.format(
        'categorical', 'linear support vector machine classifier')
)


plugin.visualizers.register_function(
    function=classify_SVC,
    inputs=inputs,
    parameters={**parameters, **svm_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **svm_parameter_descriptions},
    name='Support vector machine classifier',
    description=description.format(
        'categorical', 'support vector machine classifier')
)


plugin.visualizers.register_function(
    function=regress_linearSVR,
    inputs=inputs,
    parameters=parameters,
    input_descriptions=input_descriptions,
    parameter_descriptions={**parameter_descriptions, },
    name='Linear support vector machine regressor',
    description=description.format(
        'continuous', 'linear support vector machine regressor')
)


plugin.visualizers.register_function(
    function=regress_SVR,
    inputs=inputs,
    parameters={**parameters, **svm_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **svm_parameter_descriptions},
    name='Support vector machine regressor',
    description=description.format(
        'continuous', 'support vector machine regressor')
)


plugin.visualizers.register_function(
    function=regress_ridge,
    inputs=inputs,
    parameters={**parameters, 'solver': Str % Choices([
        'auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag', 'saga'])},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions,
        "solver": ('Solver to use in computational routines. "auto" chooses '
                   'the solver automatically based on the type of data. For '
                   'details see http://scikit-learn.org/dev/modules/generated/'
                   'sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge'
                  )},
    name='Ridge regression',
    description=description.format(
        'continuous', 'Ridge regression model')
)


plugin.visualizers.register_function(
    function=classify_kneighbors,
    inputs=inputs,
    parameters={**parameters, **neighbors_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **neighbors_parameter_descriptions},
    name='K-nearest neighbors vote classifier',
    description=description.format(
        'categorical', 'K-nearest neighbors vote classifier')
)
