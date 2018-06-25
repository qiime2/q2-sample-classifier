# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

import qiime2
import pandas as pd
import biom

from .utilities import (split_optimize_classify, _visualize, _load_data,
                        _maz_score, _visualize_maturity_index,
                        _set_parameters_and_estimator, _prepare_training_data,
                        _disable_feature_selection, _select_estimator,
                        nested_cross_validation, _fit_estimator,
                        _map_params_to_pipeline, _extract_features,
                        _plot_accuracy)


defaults = {
    'test_size': 0.2,
    'step': 0.05,
    'cv': 5,
    'n_jobs': 1,
    'n_estimators': 100,
    'estimator_c': 'RandomForestClassifier',
    'estimator_r': 'RandomForestRegressor',
    'palette': 'sirocco',
    'missing_samples': 'error'
}


def classify_samples(output_dir: str, table: biom.Table,
                     metadata: qiime2.CategoricalMetadataColumn,
                     test_size: float=defaults['test_size'],
                     step: float=defaults['step'],
                     cv: int=defaults['cv'], random_state: int=None,
                     n_jobs: int=defaults['n_jobs'],
                     n_estimators: int=defaults['n_estimators'],
                     estimator: str=defaults['estimator_c'],
                     optimize_feature_selection: bool=False,
                     parameter_tuning: bool=False,
                     palette: str=defaults['palette'],
                     missing_samples: str=defaults['missing_samples']) -> None:

    # extract column name from CategoricalMetadataColumn
    column = metadata.name

    # disable feature selection for unsupported estimators
    optimize_feature_selection, calc_feature_importance = \
        _disable_feature_selection(estimator, optimize_feature_selection)

    # specify parameters and distributions to sample from for parameter tuning
    estimator, param_dist, parameter_tuning = _set_parameters_and_estimator(
        estimator, table, metadata, column, n_estimators, n_jobs, cv,
        random_state, parameter_tuning, classification=True,
        missing_samples=missing_samples)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, column, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=calc_feature_importance, palette=palette,
        missing_samples=missing_samples)

    _visualize(output_dir, estimator, cm, importances,
               optimize_feature_selection, title='classification predictions')


def fit_classifier(table: biom.Table,
                   metadata: qiime2.CategoricalMetadataColumn,
                   step: float=defaults['step'], cv: int=defaults['cv'],
                   random_state: int=None, n_jobs: int=defaults['n_jobs'],
                   n_estimators: int=defaults['n_estimators'],
                   estimator: str=defaults['estimator_c'],
                   optimize_feature_selection: bool=False,
                   parameter_tuning: bool=False,
                   missing_samples: str=defaults['missing_samples']
                   ) -> (Pipeline, pd.DataFrame):
    estimator, importance = _fit_estimator(
        table, metadata, estimator, n_estimators, step, cv, random_state,
        n_jobs, optimize_feature_selection, parameter_tuning,
        missing_samples=missing_samples, classification=True)

    return estimator, importance


def fit_regressor(table: biom.Table,
                  metadata: qiime2.CategoricalMetadataColumn,
                  step: float=defaults['step'], cv: int=defaults['cv'],
                  random_state: int=None, n_jobs: int=defaults['n_jobs'],
                  n_estimators: int=defaults['n_estimators'],
                  estimator: str=defaults['estimator_r'],
                  optimize_feature_selection: bool=False,
                  parameter_tuning: bool=False,
                  missing_samples: str=defaults['missing_samples']
                  ) -> (Pipeline, pd.DataFrame):
    estimator, importance = _fit_estimator(
        table, metadata, estimator, n_estimators, step, cv, random_state,
        n_jobs, optimize_feature_selection, parameter_tuning,
        missing_samples=missing_samples, classification=False)

    return estimator, importance


def regress_samples(output_dir: str, table: biom.Table,
                    metadata: qiime2.NumericMetadataColumn,
                    test_size: float=defaults['test_size'],
                    step: float=defaults['step'],
                    cv: int=defaults['cv'], random_state: int=None,
                    n_jobs: int=defaults['n_jobs'],
                    n_estimators: int=defaults['n_estimators'],
                    estimator: str=defaults['estimator_r'],
                    optimize_feature_selection: bool=False,
                    stratify: str=False, parameter_tuning: bool=False,
                    missing_samples: str=defaults['missing_samples']) -> None:

    # extract column name from NumericMetadataColumn
    column = metadata.name

    # disable feature selection for unsupported estimators
    optimize_feature_selection, calc_feature_importance = \
        _disable_feature_selection(estimator, optimize_feature_selection)

    # specify parameters and distributions to sample from for parameter tuning
    estimator, param_dist, parameter_tuning = _set_parameters_and_estimator(
        estimator, table, metadata, column, n_estimators, n_jobs, cv,
        random_state, parameter_tuning, classification=True,
        missing_samples=missing_samples)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, column, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=calc_feature_importance,
        scoring=mean_squared_error, stratify=stratify, classification=False,
        missing_samples=missing_samples)

    _visualize(output_dir, estimator, cm, importances,
               optimize_feature_selection, title='regression predictions')


def predict(table: biom.Table, sample_estimator: Pipeline,
            n_jobs: int=defaults['n_jobs']) -> pd.Series:
    # extract feature data from biom
    feature_data = _extract_features(table)

    # reset n_jobs if this is a valid parameter for the estimator
    if 'est__n_jobs' in sample_estimator.get_params().keys():
        sample_estimator.set_params(est__n_jobs=n_jobs)

    # predict values and output as series
    y_pred = sample_estimator.predict(feature_data)
    # need to flatten arrays that come out as multidimensional
    y_pred = y_pred.flatten()
    y_pred = pd.Series(y_pred, index=table.ids(), name='prediction')
    y_pred.index.name = 'SampleID'

    return y_pred


def split_table(table: biom.Table, metadata: qiime2.MetadataColumn,
                test_size: float=defaults['test_size'], random_state: int=None,
                stratify: str=True,
                missing_samples: str=defaults['missing_samples']
                ) -> (biom.Table, biom.Table):
    column = metadata.name
    X_train, X_test, y_train, y_test = _prepare_training_data(
        table, metadata, column, test_size, random_state, load_data=True,
        stratify=True, missing_samples=missing_samples)
    # TODO: we can consider returning the metadata (y_train, y_test) if a
    # SampleData[Metadata] type comes into existence. For now we will just
    # throw this out.
    return X_train, X_test


def regress_samples_ncv(
        table: biom.Table, metadata: qiime2.NumericMetadataColumn,
        cv: int=defaults['cv'], random_state: int=None,
        n_jobs: int=defaults['n_jobs'],
        n_estimators: int=defaults['n_estimators'],
        estimator: str=defaults['estimator_r'], stratify: str=False,
        parameter_tuning: bool=False,
        missing_samples: str=defaults['missing_samples']
        ) -> (pd.Series, pd.DataFrame):

    y_pred, importances = nested_cross_validation(
        table, metadata, cv, random_state, n_jobs, n_estimators, estimator,
        stratify, parameter_tuning, classification=False,
        scoring=mean_squared_error, missing_samples=missing_samples)
    return y_pred, importances


def classify_samples_ncv(
        table: biom.Table, metadata: qiime2.CategoricalMetadataColumn,
        cv: int=defaults['cv'], random_state: int=None,
        n_jobs: int=defaults['n_jobs'],
        n_estimators: int=defaults['n_estimators'],
        estimator: str=defaults['estimator_c'],
        parameter_tuning: bool=False,
        missing_samples: str=defaults['missing_samples']
        ) -> (pd.Series, pd.DataFrame):

    y_pred, importances = nested_cross_validation(
        table, metadata, cv, random_state, n_jobs, n_estimators, estimator,
        stratify=True, parameter_tuning=parameter_tuning, classification=False,
        scoring=accuracy_score, missing_samples=missing_samples)
    return y_pred, importances


def scatterplot(output_dir: str, predictions: pd.Series,
                truth: qiime2.NumericMetadataColumn,
                missing_samples: str=defaults['missing_samples']) -> None:
    predictions = pd.to_numeric(predictions)
    _plot_accuracy(output_dir, predictions, truth, missing_samples,
                   classification=False, palette=None,
                   plot_title='regression scatterplot')


def confusion_matrix(output_dir: str, predictions: pd.Series,
                     truth: qiime2.CategoricalMetadataColumn,
                     missing_samples: str=defaults['missing_samples'],
                     palette: str=defaults['palette']) -> None:
    _plot_accuracy(output_dir, predictions, truth, missing_samples,
                   classification=True, palette=palette,
                   plot_title='confusion matrix')


def maturity_index(output_dir: str, table: biom.Table,
                   metadata: qiime2.Metadata, column: str, group_by: str,
                   control: str, estimator: str=defaults['estimator_r'],
                   n_estimators: int=defaults['n_estimators'],
                   test_size: float=defaults['test_size'],
                   step: float=defaults['step'], cv: int=defaults['cv'],
                   random_state: int=None,
                   n_jobs: int=defaults['n_jobs'], parameter_tuning: bool=True,
                   optimize_feature_selection: bool=True, stratify: str=False,
                   maz_stats: bool=True,
                   missing_samples: str=defaults['missing_samples']) -> None:

    # select estimator
    param_dist, estimator = _select_estimator(estimator, n_jobs, n_estimators)
    estimator = Pipeline([('dv', DictVectorizer()), ('est', estimator)])
    param_dist = _map_params_to_pipeline(param_dist)

    # split input data into control and treatment groups
    table, metadata = _load_data(
        table, metadata, missing_samples=missing_samples)
    fancy_index = metadata[group_by] == control
    md_control = metadata[fancy_index]
    table_control = [t for t, f in zip(table, fancy_index) if f]

    # train model on control data
    estimator, cm, accuracy, importances = split_optimize_classify(
        table_control, md_control, column, estimator, output_dir,
        random_state=random_state, n_jobs=n_jobs, test_size=test_size,
        step=step, cv=cv, parameter_tuning=parameter_tuning,
        optimize_feature_selection=optimize_feature_selection,
        param_dist=param_dist, calc_feature_importance=True, load_data=False,
        scoring=mean_squared_error, stratify=stratify, classification=False,
        missing_samples='ignore')

    # predict treatment data
    index = importances.index
    table = [{k: r[k] for k in r.keys() & index} for r in table]
    y_pred = estimator.predict(table)
    predicted_column = 'predicted {0}'.format(column)
    metadata[predicted_column] = y_pred

    # calculate MAZ score
    metadata = _maz_score(
        metadata, predicted_column, column, group_by, control)

    # visualize
    table = estimator.named_steps.dv.transform(table).todense()
    table = pd.DataFrame(table, index=metadata.index,
                         columns=estimator.named_steps.dv.get_feature_names())
    _visualize_maturity_index(table, metadata, group_by, column,
                              predicted_column, importances, estimator,
                              accuracy, output_dir, maz_stats=maz_stats)


# The following method is experimental and is not registered in the current
# release. Any use of the API is at user's own risk.
def detect_outliers(table: biom.Table,
                    metadata: qiime2.Metadata, subset_column: str=None,
                    subset_value: str=None,
                    n_estimators: int=defaults['n_estimators'],
                    contamination: float=0.05, random_state: int=None,
                    n_jobs: int=defaults['n_jobs'],
                    missing_samples: str='ignore') -> (pd.Series):

    features, sample_md = _load_data(
        table, metadata, missing_samples=missing_samples)

    # if opting to train on a subset, choose subset that fits criteria
    if subset_column and subset_value:
        X_train = \
            [f for s, f in
             zip(sample_md[subset_column] == subset_value, features) if s]
    # raise error if subset_column or subset_value (but not both) are set
    elif subset_column is not None or subset_value is not None:
        raise ValueError((
            'subset_column and subset_value must both be provided with a '
            'valid value to perform model training on a subset of data.'))
    else:
        X_train = features

    # fit isolation tree
    estimator = Pipeline(
        [('dv', DictVectorizer()),
         ('est', IsolationForest(n_jobs=n_jobs, n_estimators=n_estimators,
                                 contamination=contamination,
                                 random_state=random_state))])
    estimator.fit(X_train)

    # predict outlier status
    y_pred = estimator.predict(features)
    y_pred = pd.Series(y_pred, index=sample_md.index)
    # predict reports whether sample is an inlier; change to outlier status
    y_pred[y_pred == -1] = 'True'
    y_pred[y_pred == 1] = 'False'
    y_pred.name = "outlier"
    return y_pred
