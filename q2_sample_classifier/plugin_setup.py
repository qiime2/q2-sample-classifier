#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from qiime2.plugin import (
    Int, Str, Float, Range, Bool, Plugin, Metadata, Choices)
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.sample_data import SampleData
from .classify import (
    classify_random_forest, regress_random_forest, classify_linearSVC,
    regress_SVR, classify_SVC, classify_kneighbors,
    regress_ridge, regress_lasso, regress_elasticnet,
    regress_kneighbors, classify_extra_trees, classify_adaboost,
    classify_gradient_boosting, regress_extra_trees, regress_adaboost,
    regress_gradient_boosting, maturity_index, detect_outliers,
    predict_coordinates)
import q2_sample_classifier
from qiime2.plugin import SemanticType
import qiime2.plugin.model as model
import pandas as pd
import qiime2


plugin = Plugin(
    name='sample-classifier',
    version=q2_sample_classifier.__version__,
    website="https://github.com/nbokulich/q2-sample-classifier",
    package='q2_sample_classifier',
    short_description='Plugin for machine learning prediction of sample data.'
)

Coordinates = SemanticType('Coordinates', variant_of=SampleData.field['type'])


class CoordinatesFormat(model.TextFileFormat):
    def sniff(self):
        with self.open() as fh:
            for line, _ in zip(fh, range(10)):
                cells = line.split('\t')
                if len(cells) < 2:
                    return False
            return True


CoordinatesDirectoryFormat = model.SingleFileDirectoryFormat(
    'CoordinatesDirectoryFormat', 'coordinates.tsv',
    CoordinatesFormat)


def _read_dataframe(fh):
    # Using `dtype=object` and `set_index` to avoid type casting/inference
    # of any columns or the index.
    df = pd.read_csv(fh, sep='\t', header=0, dtype=object)
    df.set_index(df.columns[0], drop=True, append=False, inplace=True)
    df.index.name = None
    return df


@plugin.register_transformer
def _1(data: pd.DataFrame) -> (CoordinatesFormat):
    ff = CoordinatesFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True)
    return ff


@plugin.register_transformer
def _2(ff: CoordinatesFormat) -> (pd.DataFrame):
    with ff.open() as fh:
        df = _read_dataframe(fh)
        return df.apply(lambda x: pd.to_numeric(x, errors='ignore'))


@plugin.register_transformer
def _3(ff: CoordinatesFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh))


BooleanSeries = SemanticType('BooleanSeries', variant_of=SampleData.field['type'])


class BooleanSeriesFormat(model.TextFileFormat):
    def sniff(self):
        with self.open() as fh:
            line = fh.readline()
            for line, _ in zip(fh, range(5)):
                cells = line.strip().split('\t')
                if len(cells) != 2 or str(cells[1]) not in ('True', 'False'):
                    return False
            return True


BooleanSeriesDirectoryFormat = model.SingleFileDirectoryFormat(
    'BooleanSeriesDirectoryFormat', 'outliers.tsv',
    BooleanSeriesFormat)


@plugin.register_transformer
def _4(data: pd.Series) -> (BooleanSeriesFormat):
    ff = BooleanSeriesFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True)
    return ff


@plugin.register_transformer
def _5(ff: BooleanSeriesFormat) -> (pd.Series):
    with ff.open() as fh:
        return _read_dataframe(fh)


@plugin.register_transformer
def _6(ff: BooleanSeriesFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh))


plugin.register_formats(CoordinatesFormat, CoordinatesDirectoryFormat,
                        BooleanSeriesFormat, BooleanSeriesDirectoryFormat)

plugin.register_semantic_types(Coordinates, BooleanSeries)

plugin.register_semantic_type_to_format(
    SampleData[Coordinates],
    artifact_format=CoordinatesDirectoryFormat)

plugin.register_semantic_type_to_format(
    SampleData[BooleanSeries],
    artifact_format=BooleanSeriesDirectoryFormat)

description = ('Predict {0} sample metadata classes using a {1}. Splits input '
               'data into training and test sets. The training set is used '
               'to train and test the classifier using a stratified k-fold '
               'cross-validation scheme. This includes optional steps for '
               'automated feature extraction and hyperparameter optimization. '
               'The test set validates classification accuracy of the '
               'optimized estimator. Outputs classification results for test '
               'set, and optionally a trained estimator to use on additional '
               'unknown samples. For more details on the learning algorithm, '
               'see {2}')

inputs = {'table': FeatureTable[Frequency]}

base_parameters = {'metadata': Metadata,
                   'random_state': Int,
                   'n_jobs': Int}

regressor_parameters = {
    'stratify': Bool,
}

parameters = {**base_parameters,
              'category': Str,
              'test_size': Float % Range(0.0, 1.0, inclusive_end=False,
                                         inclusive_start=False),
              'step': Float % Range(0.0, 1.0, inclusive_end=False,
                                    inclusive_start=False),
              'cv': Int % Range(1, None),
              'parameter_tuning': Bool,
              'optimize_feature_selection': Bool}

input_descriptions = {'table': ('Feature table containing all features that '
                                'should be used for target prediction.')}

base_parameter_descriptions = {
    'metadata': 'Sample metadata to use as prediction targets.',
    'random_state': 'Seed used by random number generator.',
    'n_jobs': 'Number of jobs to run in parallel.',
}

parameter_descriptions = {
    **base_parameter_descriptions,
    'category': 'Metadata category to use for training and prediction.',
    'test_size': ('Fraction of input samples to exclude from training set '
                  'and use for classifier testing.'),
    'step': ('If optimize_feature_selection is True, step is the '
             'percentage of features to remove at each iteration.'),
    'cv': 'Number of k-fold cross-validations to perform.',
    'parameter_tuning': ('Automatically tune hyperparameters using random '
                         'grid search?'),
    'optimize_feature_selection': ('Automatically optimize input feature '
                                   'selection using recursive feature '
                                   'elimination?')
}

regressor_parameter_descriptions = {
    'stratify': ('Evenly stratify training and test data among metadata '
                 'categories. If True, all values in category must match at '
                 'least two samples.'),
}

ensemble_parameters = {
    'n_estimators': Int % Range(1, None)
}


ensemble_parameter_descriptions = {
    'n_estimators': ('Number of trees to grow for estimation. More trees will '
                     'improve predictive accuracy up to a threshold level, '
                     'but will also increase time and memory requirements.'),
}


svm_parameters = {
    'kernel': Str % Choices(['linear', 'poly', 'rbf', 'sigmoid'])
}


svm_parameter_descriptions = {
    'kernel': 'Specifies the kernel type to be used in the algorithm.'
}


neighbors_parameters = {
    **{k: parameters[k] for k in parameters.keys() if
       k != "optimize_feature_selection"},
    'algorithm': Str % Choices(['ball_tree', 'kd_tree', 'brute', 'auto'])
}


neighbors_parameter_descriptions = {
    **{k: parameter_descriptions[k] for k in parameter_descriptions.keys() if
       k != "optimize_feature_selection"},
    'algorithm': ('Algorithm used to compute the nearest neighbors. Default, '
                  'auto, will attempt to decide the most appropriate '
                  'algorithm based on the values passed to fit method.')
    }


plugin.visualizers.register_function(
    function=classify_random_forest,
    inputs=inputs,
    parameters={**parameters, **ensemble_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **ensemble_parameter_descriptions},
    name='Random forest sample classifier',
    description=description.format(
        'categorical', 'random forest classifier',
        'http://scikit-learn.org/stable/modules/ensemble.html')
)


plugin.visualizers.register_function(
    function=classify_extra_trees,
    inputs=inputs,
    parameters={**parameters, **ensemble_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **ensemble_parameter_descriptions},
    name='Extra Trees sample classifier',
    description=description.format(
        'categorical', 'Extra Trees classifier',
        'http://scikit-learn.org/stable/modules/ensemble.html')
)


plugin.visualizers.register_function(
    function=classify_adaboost,
    inputs=inputs,
    parameters={**parameters, **ensemble_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **ensemble_parameter_descriptions},
    name='AdaBoost decision tree sample classifier',
    description=description.format(
        'categorical', 'Adaboost classifier',
        'http://scikit-learn.org/stable/modules/ensemble.html')
)


plugin.visualizers.register_function(
    function=classify_gradient_boosting,
    inputs=inputs,
    parameters={**parameters, **ensemble_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **ensemble_parameter_descriptions},
    name='Gradient boosting classifier',
    description=description.format(
        'categorical', 'Gradient boosting classifier',
        'http://scikit-learn.org/stable/modules/ensemble.html')
)


plugin.visualizers.register_function(
    function=regress_random_forest,
    inputs=inputs,
    parameters={**parameters, **ensemble_parameters, **regressor_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **ensemble_parameter_descriptions,
        **regressor_parameter_descriptions},
    name='Random forest regressor',
    description=description.format(
        'continuous', 'random forest regressor',
        'http://scikit-learn.org/stable/modules/ensemble.html')
)


plugin.visualizers.register_function(
    function=regress_extra_trees,
    inputs=inputs,
    parameters={**parameters, **ensemble_parameters, **regressor_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **ensemble_parameter_descriptions,
        **regressor_parameter_descriptions},
    name='Extra Trees regressor',
    description=description.format(
        'continuous', 'Extra Trees regressor',
        'http://scikit-learn.org/stable/modules/ensemble.html')
)


plugin.visualizers.register_function(
    function=regress_adaboost,
    inputs=inputs,
    parameters={**parameters, **ensemble_parameters, **regressor_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **ensemble_parameter_descriptions,
        **regressor_parameter_descriptions},
    name='AdaBoost decision tree regressor',
    description=description.format(
        'continuous', 'Adaboost regressor',
        'http://scikit-learn.org/stable/modules/ensemble.html')
)


plugin.visualizers.register_function(
    function=regress_gradient_boosting,
    inputs=inputs,
    parameters={**parameters, **ensemble_parameters, **regressor_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **ensemble_parameter_descriptions,
        **regressor_parameter_descriptions},
    name='Gradient boosting regressor',
    description=description.format(
        'continuous', 'Gradient boosting regressor',
        'http://scikit-learn.org/stable/modules/ensemble.html')
)


plugin.visualizers.register_function(
    function=classify_linearSVC,
    inputs=inputs,
    parameters=parameters,
    input_descriptions=input_descriptions,
    parameter_descriptions=parameter_descriptions,
    name='Linear support vector machine classifier',
    description=description.format(
        'categorical', 'linear support vector machine classifier',
        'http://scikit-learn.org/dev/modules/svm.html')
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
        'categorical', 'support vector machine classifier',
        'http://scikit-learn.org/dev/modules/svm.html')
)


plugin.visualizers.register_function(
    function=regress_SVR,
    inputs=inputs,
    parameters={**parameters, **svm_parameters, **regressor_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **svm_parameter_descriptions,
        **regressor_parameter_descriptions},
    name='Support vector machine regressor',
    description=description.format(
        'continuous', 'support vector machine regressor',
        'http://scikit-learn.org/dev/modules/svm.html')
)


plugin.visualizers.register_function(
    function=regress_ridge,
    inputs=inputs,
    parameters={
        **parameters, **regressor_parameters, 'solver': Str % Choices([
            'auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag', 'saga'])},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions,
        **regressor_parameter_descriptions,
        "solver": ('Solver to use in computational routines. "auto" chooses '
                   'the solver automatically based on the type of data. For '
                   'details see http://scikit-learn.org/dev/modules/generated/'
                   'sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge'
                   )},
    name='Ridge regression',
    description=description.format(
        'continuous', 'Ridge linear regression',
        'http://scikit-learn.org/dev/modules/linear_model.html')
)


plugin.visualizers.register_function(
    function=regress_lasso,
    inputs=inputs,
    parameters={**parameters, **regressor_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **regressor_parameter_descriptions},
    name='Lasso regression',
    description=description.format(
        'continuous', 'Lasso linear regression',
        'http://scikit-learn.org/dev/modules/linear_model.html')
)


plugin.visualizers.register_function(
    function=regress_elasticnet,
    inputs=inputs,
    parameters={**parameters, **regressor_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions, **regressor_parameter_descriptions},
    name='Elastic Net regression',
    description=description.format(
        'continuous', 'Elastic Net linear regression',
        'http://scikit-learn.org/dev/modules/linear_model.html')
)


plugin.visualizers.register_function(
    function=regress_kneighbors,
    inputs=inputs,
    parameters={**neighbors_parameters, **regressor_parameters},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **neighbors_parameter_descriptions,
        **regressor_parameter_descriptions},
    name='K-nearest neighbors regression',
    description=description.format(
        'continuous', 'K-nearest neighbors regression',
        'http://scikit-learn.org/dev/modules/neighbors.html')
)


plugin.visualizers.register_function(
    function=classify_kneighbors,
    inputs=inputs,
    parameters=neighbors_parameters,
    input_descriptions=input_descriptions,
    parameter_descriptions=neighbors_parameter_descriptions,
    name='K-nearest neighbors vote classifier',
    description=description.format(
        'categorical', 'K-nearest neighbors vote classifier',
        'http://scikit-learn.org/dev/modules/neighbors.html')
)


plugin.visualizers.register_function(
    function=maturity_index,
    inputs=inputs,
    parameters={'group_by': Str,
                'control': Str,
                'estimator': Str % Choices([
                    'RandomForestRegressor', 'ExtraTreesRegressor',
                    'GradientBoostingRegressor', 'SVR', 'Ridge', 'Lasso',
                    'ElasticNet']),
                **parameters,
                **ensemble_parameters,
                **regressor_parameters,
                'maz_stats': Bool,
                },
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions,
        'group_by': ('Metadata category to use for plotting and significance '
                     'testing between main treatment groups.'),
        'control': (
            'Value of group_by to use as control group. The regression  model '
            'will be trained using only control group data, and the maturity '
            'scores of other groups consequently will be assessed relative to '
            'this group.'),
        'estimator': 'Regression model to use for prediction.',
        **ensemble_parameter_descriptions,
        **regressor_parameter_descriptions,
        'maz_stats': 'Calculate anova and pairwise tests on MAZ scores?',
    },
    name='Microbial maturity index prediction',
    description=('Calculates a "microbial maturity" index from a regression '
                 'model trained on feature data to predict a given continuous '
                 'metadata category, e.g., to predict age as a function of '
                 'microbiota composition. The model is trained on a subset of '
                 'control group samples, then predicts the category value for '
                 'all samples. This visualization computes maturity index '
                 'z-scores to compare relative "maturity" between each group, '
                 'as described in doi:10.1038/nature13421. This method can '
                 'be used to predict between-group differences in relative '
                 'trajectory across any type of continuous metadata gradient, '
                 'e.g., intestinal microbiome development by age, microbial '
                 'succession during wine fermentation, or microbial community '
                 'differences along environmental gradients, as a function of '
                 'two or more different "treatment" groups.')
)


plugin.methods.register_function(
    function=detect_outliers,
    inputs=inputs,
    parameters={**base_parameters,
                'subset_category': Str,
                'subset_value': Str,
                'contamination': Float % Range(0.0, 0.5, inclusive_end=True,
                                               inclusive_start=True),
                **ensemble_parameters,
                },
    outputs=[('inliers', SampleData[BooleanSeries])],
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **base_parameter_descriptions,
        'subset_category': ('Metadata category to use for selecting sample '
                            'subset for training the decision function.'),
        'subset_value': 'Value of subset_category to use as control group.',
        'contamination': ('The amount of expected contamination of the data '
                          'set, i.e., the proportion of outliers in the data '
                          'set. Used when fitting to define the threshold on '
                          'the decision function.'),
        **ensemble_parameter_descriptions,
    },
    output_descriptions={
        'inliers': ('Vector containing inlier status of each input sample. '
                    'Inliers have value 1, outliers have value -1.')
    },
    name='Predict dataset outliers and contaminants',
    description=(
        'Detect outlier samples within a given sample class. Applications '
        'include but are not limited to detecting potentially contaminated '
        'samples, detecting potentially mislabeled samples, and detecting '
        'significant novelty, e.g., patients who responded to a treatment.\n\n'
        'Input a feature table, possibly filtered to remove samples, '
        'depending on the goals of this analysis. Outliers can be detected '
        'from multiple sample types simultaneously, provided the goal is not '
        'to detect mislabeled samples or samples cross-contaminated with '
        'another sample type in this table. E.g., for detecting novelty or '
        'exogenous contaminants (e.g., from reagents), many different sample '
        'types may be tested simultaneously. Otherwise, the feature table '
        'should be filtered to contain only one or more sample classes '
        'between which cross-contamination is not suspected, or if these '
        'sample classes are highly resolved and mislabeled samples are not '
        'suspected. These assumptions may be supported by a preliminary '
        'principal coordinates analysis or other diversity analyses to '
        'determine how well resolved sample classes are and whether some '
        'sample classes appear to cluster with the wrong class(es).\n\n'
        'Inputs support two different modes: if subset_category and '
        'subset_value are set, a subset of the input table is used as a '
        '"gold standard" sample pool for training the model. This mode is '
        'useful, for example, if you have a subset of "positive control" '
        'samples that represent the known diversity of your sample types. '
        'Otherwise, the model is trained on all samples. Regardless of the '
        'input mode used, outlier status is predicted on all samples.\n\n'
        'Returns a series of values documenting outlier status: inliers have '
        'value of 1, outliers have value of -1. This series may be used to '
        'filter a feature table, if appropriate, using '
        'q2_feature_table.filter_samples, to remove contaminants or focus on '
        'novelty samples.\n\nIf interested in potentially mislabeled samples, '
        'use a sample classifier in q2_sample_classifier or principal '
        'coordinates analysis to determine whether outliers classify as or '
        'cluster with another sample type.\n\nFor more information on the '
        'underlying isolation forest model, see '
        'http://scikit-learn.org/stable/modules/outlier_detection.html')
)


plugin.methods.register_function(
    function=predict_coordinates,
    inputs=inputs,
    parameters={
        **{k: parameters[k] for k in parameters.keys() if k != "category"},
        'latitude': Str,
        'longitude': Str,
        'estimator': Str % Choices([
            'RandomForestRegressor', 'ExtraTreesRegressor', 'Lasso',
            'GradientBoostingRegressor', 'SVR', 'Ridge', 'ElasticNet']),
        **ensemble_parameters,
    },
    outputs=[('predictions', SampleData[Coordinates]),
             ('prediction_regression', SampleData[Coordinates])],
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **{k: parameter_descriptions[k] for k in parameter_descriptions.keys()
           if k != "category"},
        'latitude': ('Category name containing latitude or first dimension '
                     'coordinates in sample metadata file.'),
        'longitude': ('Category name containing longitude or second dimension '
                      'coordinates in sample metadata file.'),
        'estimator': 'Regression model to use for prediction.',
        **ensemble_parameter_descriptions,
    },
    output_descriptions={
        'predictions': 'Predicted coordinates for each dimension.',
        'prediction_regression': 'Regression results for each dimension.',
    },
    name='Predict sample geocoordinates.',
    description=(
        'Predict two-dimensional coordinates as a function of microbiota '
        'composition. E.g., this function could be used to predict '
        'latitude and longitude (2-D) or precise location within any 2-D '
        'physical space, such as the built environment. Metadata '
        'must be in float format, e.g., decimal degrees geocoordinates. '
        'Ouput consists of predicted coordinates, accuracy scores for each '
        'dimension, and linear regression results for each dimension.')
)
