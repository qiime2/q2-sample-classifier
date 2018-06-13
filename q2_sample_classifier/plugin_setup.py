# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import (
    Int, Str, Float, Range, Bool, Plugin, Metadata, Choices, MetadataColumn,
    Numeric, Categorical, SemanticType, Citations, ValidationError)
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData
from .classify import (
    classify_samples, regress_samples, maturity_index, regress_samples_ncv,
    classify_samples_ncv)
from .visuals import _custom_palettes
import q2_sample_classifier
import qiime2.plugin.model as model
import pandas as pd
import qiime2


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


BooleanSeries = SemanticType(
    'BooleanSeries', variant_of=SampleData.field['type'])


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


Predictions = SemanticType(
    'Predictions', variant_of=SampleData.field['type'])


class PredictionsFormat(model.TextFileFormat):
    def _validate(self, n_records=None):
        with self.open() as fh:
            # validate header
            # for now we will not validate any information in the header,
            # since the name of the predicted column should be flexible. The
            # header name written by methods in q2-sample-classifier will be
            # "predicted-*", but this should also accommodate user-defined
            # column names.
            line = fh.readline()

            # validate body
            has_data = False
            for line_number, line in enumerate(fh, start=2):
                cells = line.strip().split('\t')
                if len(cells) != 2:
                    raise ValidationError(
                        "Expected data record to be TSV with two "
                        "fields. Detected {0} fields at line {1}:\n\n{2!r}"
                        .format(len(cells), line_number, cells))
                has_data = True
                if n_records is not None and (line_number - 1) >= n_records:
                    break

            if not has_data:
                raise ValidationError(
                    "There must be at least one data record present in the "
                    "file in addition to the header line.")
            return True

    def _validate_(self, level):
        record_count_map = {'min': 5, 'max': None}
        self._validate(record_count_map[level])


PredictionsDirectoryFormat = model.SingleFileDirectoryFormat(
    'PredictionsDirectoryFormat', 'predictions.tsv',
    PredictionsFormat)


Importance = SemanticType(
    'Importance', variant_of=FeatureData.field['type'])


class ImportanceFormat(model.TextFileFormat):
    def _validate(self, n_records=None):
        with self.open() as fh:
            # validate header
            # for now we will not validate any information in the header,
            # since column names, count etc are frequently unique to individual
            # estimators. Let's keep this flexible.
            line = fh.readline()

            # validate body
            has_data = False
            for line_number, line in enumerate(fh, start=2):
                cells = line.strip().split('\t')
                if len(cells) < 2:
                    raise ValidationError(
                        "Expected data record to be TSV with two or more "
                        "fields. Detected {0} fields at line {1}:\n\n{2!r}"
                        .format(len(cells), line_number, cells))
                # all values (except row name) should be numbers
                try:
                    [float(c) for c in cells[1:]]
                except ValueError:
                    raise ValidationError(
                        "Columns must contain only numeric values. "
                        "A non-numeric value ({0!r}) was detected at line "
                        "{1}.".format(cells[1], line_number))

                has_data = True
                if n_records is not None and (line_number - 1) >= n_records:
                    break

            if not has_data:
                raise ValidationError(
                    "There must be at least one data record present in the "
                    "file in addition to the header line.")
            return True

    def _validate_(self, level):
        record_count_map = {'min': 5, 'max': None}
        self._validate(record_count_map[level])


ImportanceDirectoryFormat = model.SingleFileDirectoryFormat(
    'ImportanceDirectoryFormat', 'importance.tsv',
    ImportanceFormat)


def _read_dataframe(fh):
    # Using `dtype=object` and `set_index` to avoid type casting/inference
    # of any columns or the index.
    df = pd.read_csv(fh, sep='\t', header=0, dtype='str')
    df.set_index(df.columns[0], drop=True, append=False, inplace=True)
    df.index.name = 'id'
    return df


@plugin.register_transformer
def _1(data: pd.Series) -> (BooleanSeriesFormat):
    ff = BooleanSeriesFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True)
    return ff


@plugin.register_transformer
def _2(ff: BooleanSeriesFormat) -> (pd.Series):
    with ff.open() as fh:
        df = _read_dataframe(fh)
        return df.iloc[:, 0]


@plugin.register_transformer
def _3(ff: BooleanSeriesFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh))


@plugin.register_transformer
def _4(data: pd.Series) -> (PredictionsFormat):
    ff = PredictionsFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True)
    return ff


@plugin.register_transformer
def _5(ff: PredictionsFormat) -> (pd.Series):
    with ff.open() as fh:
        df = _read_dataframe(fh)
        return df.iloc[:, 0]


@plugin.register_transformer
def _6(ff: PredictionsFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh))


@plugin.register_transformer
def _7(data: pd.DataFrame) -> (ImportanceFormat):
    ff = ImportanceFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True)
    return ff


@plugin.register_transformer
def _8(ff: ImportanceFormat) -> (pd.DataFrame):
    with ff.open() as fh:
        return _read_dataframe(fh)


@plugin.register_transformer
def _9(ff: ImportanceFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh))


plugin.register_formats(BooleanSeriesFormat, BooleanSeriesDirectoryFormat,
                        ImportanceFormat, ImportanceDirectoryFormat,
                        PredictionsFormat, PredictionsDirectoryFormat)

plugin.register_semantic_types(BooleanSeries, Importance, Predictions)

plugin.register_semantic_type_to_format(
    SampleData[BooleanSeries],
    artifact_format=BooleanSeriesDirectoryFormat)

plugin.register_semantic_type_to_format(
    SampleData[Predictions],
    artifact_format=PredictionsDirectoryFormat)

plugin.register_semantic_type_to_format(
    FeatureData[Importance],
    artifact_format=ImportanceDirectoryFormat)


description = ('Predicts a {0} sample metadata column using a {1}. Splits '
               'input data into training and test sets. The training set is '
               'used to train and test the estimator using a stratified '
               'k-fold cross-validation scheme. This includes optional steps '
               'for automated feature extraction and hyperparameter '
               'optimization. The test set validates classification accuracy '
               'of the optimized estimator. Outputs classification results '
               'for test set. For more details on the learning algorithm, '
               'see http://scikit-learn.org/stable/supervised_learning.html')

cv_description = ('Predicts a {0} sample metadata column using a {1}. Uses '
                  'nested stratified k-fold cross validation for automated '
                  'hyperparameter optimization and sample prediction. Outputs '
                  'predicted values for each input sample, and relative '
                  'importance of each feature for model accuracy. For more '
                  'details on the learning algorithm, see '
                  'http://scikit-learn.org/stable/supervised_learning.html')

inputs = {'table': FeatureTable[Frequency]}

input_descriptions = {'table': ('Feature table containing all features that '
                                'should be used for target prediction.')}

parameters = {
    'base': {
        'random_state': Int,
        'n_jobs': Int,
        'n_estimators': Int % Range(1, None)},
    'standard': {
        'test_size': Float % Range(0.0, 1.0, inclusive_end=False,
                                   inclusive_start=False),
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
                'Forest, AdaBoost, ExtraTrees, and GradientBoosting.')},
    'standard': {
        'test_size': ('Fraction of input samples to exclude from training set '
                      'and use for classifier testing.'),
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
        **parameters['standard'],
        **parameters['cv'],
        'metadata': MetadataColumn[Categorical],
        'estimator': Str % Choices(
            ['RandomForestClassifier', 'ExtraTreesClassifier',
             'GradientBoostingClassifier', 'AdaBoostClassifier',
             'KNeighborsClassifier', 'LinearSVC', 'SVC']),
        'palette': Str % Choices(_custom_palettes().keys())},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['standard'],
        **parameter_descriptions['cv'],
        'metadata': ('Categorical metadata column to use as prediction '
                     'target.'),
        **parameter_descriptions['estimator'],
        'palette': 'The color palette to use for plotting.'},
    name='Supervised learning classifier.',
    description=description.format(
        'categorical', 'supervised learning classifier')
)

plugin.visualizers.register_function(
    function=regress_samples,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['standard'],
        **parameters['cv'],
        'metadata': MetadataColumn[Numeric],
        **parameters['regressor'],
        'estimator': Str % Choices(
            ['RandomForestRegressor', 'ExtraTreesRegressor',
             'GradientBoostingRegressor', 'AdaBoostRegressor', 'ElasticNet',
             'Ridge', 'Lasso', 'KNeighborsRegressor', 'LinearSVR', 'SVR'])},
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['standard'],
        **parameter_descriptions['cv'],
        **parameter_descriptions['regressor'],
        'metadata': 'Numeric metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
    name='Supervised learning regressor.',
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
        'estimator': Str % Choices(
            ['RandomForestRegressor', 'ExtraTreesRegressor',
             'GradientBoostingRegressor', 'AdaBoostRegressor', 'ElasticNet',
             'Ridge', 'Lasso', 'KNeighborsRegressor', 'LinearSVR', 'SVR'])},
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
    description=cv_description.format(
        'continuous', 'supervised learning regressor')
)

plugin.methods.register_function(
    function=classify_samples_ncv,
    inputs=inputs,
    parameters={
        **parameters['base'],
        **parameters['cv'],
        'metadata': MetadataColumn[Categorical],
        'estimator': Str % Choices(
            ['RandomForestClassifier', 'ExtraTreesClassifier',
             'GradientBoostingClassifier', 'AdaBoostClassifier',
             'KNeighborsClassifier', 'LinearSVC', 'SVC'])},
    outputs=outputs,
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['cv'],
        'metadata': 'Categorical metadata column to use as prediction target.',
        **parameter_descriptions['estimator']},
    output_descriptions=output_descriptions,
    name='Nested cross-validated supervised learning classifier.',
    description=cv_description.format(
        'categorical', 'supervised learning classifier')
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
                **parameters['base'],
                **parameters['standard'],
                **parameters['cv'],
                'metadata': Metadata,
                'column': Str,
                **parameters['regressor'],
                'maz_stats': Bool,
                },
    input_descriptions=input_descriptions,
    parameter_descriptions={
        **parameter_descriptions['base'],
        **parameter_descriptions['standard'],
        **parameter_descriptions['cv'],
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
