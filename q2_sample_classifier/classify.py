# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import qiime2
import pandas as pd
import skbio

from .utilities import (split_optimize_classify, _visualize, _load_data,
                        _maz_score, _visualize_maturity_index,
                        _set_parameters_and_estimator,
                        _disable_feature_selection, _select_estimator, _predict_and_plot)

defaults = {
    'test_size': 0.2,
    'step': 0.05,
    'cv': 5,
    'n_jobs': 1,
    'n_estimators': 100,
    'estimator_r': 'RandomForestClassifier',
    'palette': 'sirocco'
}

def classify_samples_from_dist(output_dir: str, dmtx: skbio.DistanceMatrix,
                     metadata: qiime2.CategoricalMetadataColumn,
                     palette: str=defaults['palette']) -> None:

    column = metadata.to_series()
    classifier = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
    classifier.fit(dmtx.to_data_frame(), column)
    # warning, this probably doesn't do loo (leave one out). leave zero out!
    predictions = classifier.predict(dmtx.to_data_frame())
    print(predictions)
    # writes a pdf file that's important for viz
    predictions2, confusion = _predict_and_plot(output_dir, column, predictions, classifier, .8)
    print(predictions2)
    _visualize(output_dir, classifier, predictions2, .7, None,
               False, title='classification predictions')


def classify_samples(output_dir: str, table: pd.DataFrame,
                     metadata: qiime2.CategoricalMetadataColumn,
                     test_size: float=defaults['test_size'],
                     step: float=defaults['step'],
                     cv: int=defaults['cv'], random_state: int=None,
                     n_jobs: int=defaults['n_jobs'],
                     n_estimators: int=defaults['n_estimators'],
                     estimator: str=defaults['estimator_r'],
                     optimize_feature_selection: bool=False,
                     parameter_tuning: bool=False,
                     palette: str=defaults['palette']) -> None:

    # extract column name from CategoricalMetadataColumn
    column = metadata.to_series().name

    # disable feature selection for unsupported estimators
    optimize_feature_selection, calc_feature_importance = \
        _disable_feature_selection(estimator, optimize_feature_selection)

    # specify parameters and distributions to sample from for parameter tuning
    estimator, param_dist, parameter_tuning = _set_parameters_and_estimator(
        estimator, table, metadata, column, n_estimators, n_jobs, cv,
        random_state, parameter_tuning, classification=True)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, column, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=calc_feature_importance, palette=palette)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection, title='classification predictions')


def regress_samples(output_dir: str, table: pd.DataFrame,
                    metadata: qiime2.NumericMetadataColumn,
                    test_size: float=defaults['test_size'],
                    step: float=defaults['step'],
                    cv: int=defaults['cv'], random_state: int=None,
                    n_jobs: int=defaults['n_jobs'],
                    n_estimators: int=defaults['n_estimators'],
                    estimator: str='RandomForestRegressor',
                    optimize_feature_selection: bool=False,
                    stratify: str=False, parameter_tuning: bool=False) -> None:

    # extract column name from NumericMetadataColumn
    column = metadata.to_series().name

    # disable feature selection for unsupported estimators
    optimize_feature_selection, calc_feature_importance = \
        _disable_feature_selection(estimator, optimize_feature_selection)

    # specify parameters and distributions to sample from for parameter tuning
    estimator, param_dist, parameter_tuning = _set_parameters_and_estimator(
        estimator, table, metadata, column, n_estimators, n_jobs, cv,
        random_state, parameter_tuning, classification=True)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, column, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=calc_feature_importance,
        scoring=mean_squared_error, stratify=stratify, classification=False)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection, title='regression predictions')


def maturity_index(output_dir: str, table: pd.DataFrame,
                   metadata: qiime2.Metadata, column: str, group_by: str,
                   control: str, estimator: str='RandomForestRegressor',
                   n_estimators: int=defaults['n_estimators'],
                   test_size: float=defaults['test_size'],
                   step: float=defaults['step'], cv: int=defaults['cv'],
                   random_state: int=None,
                   n_jobs: int=defaults['n_jobs'], parameter_tuning: bool=True,
                   optimize_feature_selection: bool=True, stratify: str=False,
                   maz_stats: bool=True) -> None:

    # select estimator
    param_dist, estimator = _select_estimator(estimator, n_jobs, n_estimators)

    # split input data into control and treatment groups
    table, metadata = _load_data(table, metadata)
    md_control = metadata[metadata[group_by] == control]
    table_control = table.loc[list(md_control.index.values)]

    # train model on control data
    estimator, cm, accuracy, importances = split_optimize_classify(
        table_control, md_control, column, estimator, output_dir,
        random_state=random_state, n_jobs=n_jobs, test_size=test_size,
        step=step, cv=cv, parameter_tuning=parameter_tuning,
        optimize_feature_selection=optimize_feature_selection,
        param_dist=param_dist, calc_feature_importance=True, load_data=False,
        scoring=mean_squared_error, stratify=stratify, classification=False)

    # predict treatment data
    table = table.loc[:, importances["feature"]]
    y_pred = estimator.predict(table)
    predicted_column = 'predicted {0}'.format(column)
    metadata[predicted_column] = y_pred

    # calculate MAZ score
    metadata = _maz_score(
        metadata, predicted_column, column, group_by, control)

    # visualize
    _visualize_maturity_index(table, metadata, group_by, column,
                              predicted_column, importances, estimator,
                              accuracy, output_dir, maz_stats=maz_stats)


# The following method is experimental and is not registered in the current
# release. Any use of the API is at user's own risk.
def detect_outliers(table: pd.DataFrame,
                    metadata: qiime2.Metadata, subset_column: str=None,
                    subset_value: str=None,
                    n_estimators: int=defaults['n_estimators'],
                    contamination: float=0.05, random_state: int=None,
                    n_jobs: int=defaults['n_jobs']) -> (pd.Series):

    features, sample_md = _load_data(table, metadata)

    # if opting to train on a subset, choose subset that fits criteria
    if subset_column and subset_value:
        y_train = sample_md[sample_md[subset_column] == subset_value]
        X_train = table.loc[list(y_train.index.values)]
    # raise error if subset_column or subset_value (but not both) are set
    elif subset_column is not None or subset_value is not None:
        raise ValueError((
            'subset_column and subset_value must both be provided with a '
            'valid value to perform model training on a subset of data.'))
    else:
        X_train = features

    # fit isolation tree
    estimator = IsolationForest(n_jobs=n_jobs, n_estimators=n_estimators,
                                contamination=contamination,
                                random_state=random_state)
    estimator.fit(X_train)

    # predict outlier status
    y_pred = estimator.predict(features)
    y_pred = pd.Series(y_pred, index=features.index)
    # predict reports whether sample is an inlier; change to outlier status
    y_pred[y_pred == -1] = 'True'
    y_pred[y_pred == 1] = 'False'
    y_pred.name = "outlier"
    return y_pred
