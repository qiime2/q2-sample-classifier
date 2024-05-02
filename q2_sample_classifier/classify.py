# ----------------------------------------------------------------------------
# Copyright (c) 2017-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import qiime2
from qiime2.plugin import get_available_cores
import pandas as pd
import biom
import skbio

from .utilities import (_load_data, _prepare_training_data,
                        nested_cross_validation, _fit_estimator,
                        _extract_features, _plot_accuracy,
                        _summarize_estimator, predict_probabilities,
                        _classifiers)


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


def metatable(ctx,
              metadata,
              table=None,
              missing_samples='ignore',
              missing_values='error',
              drop_all_unique=False):
    # gather numeric metadata
    metadata = metadata.filter_columns(
        column_type='numeric', drop_all_unique=drop_all_unique,
        drop_zero_variance=True, drop_all_missing=True).to_dataframe()

    if missing_values == 'drop_samples':
        metadata = metadata.dropna(axis=0)
    elif missing_values == 'drop_features':
        metadata = metadata.dropna(axis=1)
    elif missing_values == 'error' and metadata.isnull().values.any():
        raise ValueError('You are attempting to coerce metadata containing '
                         'missing values into a feature table! These may '
                         'cause fatal errors downstream and must be removed '
                         'or converted to 0. See the missing_values parameter '
                         'to review your options.')
    elif missing_values == 'fill':
        metadata = metadata.fillna(0.)

    # drop columns with negative values
    # grab column IDs with all values >= 0
    metadata = metadata.loc[:, (metadata >= 0).all(axis=0)]

    if len(metadata.columns) == 0:
        raise ValueError('All metadata columns have been filtered.')
    if len(metadata.index) == 0:
        raise ValueError('All metadata samples have been filtered.')

    # only retain IDs that intersect with table
    if table is not None:
        tab = table.view(biom.Table)
        table_ids = set(tab.ids())
        metadata_ids = set(metadata.index)
        sample_ids = table_ids.intersection(metadata_ids)
        if missing_samples == 'error' and len(sample_ids) != len(table_ids):
            raise ValueError('Missing samples in metadata: %r' %
                             table_ids.difference(metadata_ids))
        else:
            metadata = metadata.loc[list(sample_ids)]
        if len(sample_ids) < len(table_ids):
            tab = tab.filter(
                ids_to_keep=sample_ids, axis='sample', inplace=False)
            table = ctx.make_artifact('FeatureTable[Frequency]', tab)

    # convert to FeatureTable[Frequency]
    metadata = metadata.T
    metadata = biom.table.Table(
        metadata.values, metadata.index, metadata.columns)
    metatab = ctx.make_artifact('FeatureTable[Frequency]', metadata)

    # optionally merge with existing feature table
    if table is not None:
        merge = ctx.get_action('feature_table', 'merge')
        metatab, = merge(
            [table, metatab], overlap_method='error_on_overlapping_feature')

    return metatab


def _fit_predict_knn_cv(
        x: pd.DataFrame, y: pd.Series, k: int, cv: int,
        random_state: int, n_jobs: int
) -> (pd.Series, pd.Series):
    if n_jobs == 0:
        n_jobs = get_available_cores()

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # train and test with CV
    predictions, pred_ids, truth = [], [], []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index, train_index], \
                          x.iloc[test_index, train_index]
        y_train, y_test = y[train_index], y[test_index]

        knn = KNeighborsClassifier(
            n_neighbors=k, metric='precomputed', n_jobs=n_jobs
        )
        knn.fit(x_train, y_train)

        # gather predictions for the confusion matrix
        predictions.append(knn.predict(x_test))
        pred_ids.extend(x_test.index.tolist())
        truth.append(y_test)

    predictions = pd.Series(
        np.concatenate(predictions).ravel(),
        index=pd.Index(pred_ids, name='SampleID')
    )
    truth = pd.concat(truth)
    truth.index.name = 'SampleID'

    return predictions, truth


def classify_samples_from_dist(
        ctx, distance_matrix, metadata, k=1, cv=defaults['cv'],
        random_state=None, n_jobs=defaults['n_jobs'],
        palette=defaults['palette']
):
    """ Trains and evaluates a KNN classifier from a distance matrix
        using cross-validation."""
    distance_matrix = distance_matrix \
        .view(skbio.DistanceMatrix) \
        .to_data_frame()
    # reorder (required for splitting into train/test)
    metadata_ser = metadata.to_series()[distance_matrix.index]

    predictions, truth = _fit_predict_knn_cv(
        distance_matrix, metadata_ser, k, cv, random_state, n_jobs
    )
    predictions = qiime2.Artifact.import_data(
        'SampleData[ClassifierPredictions]', predictions
    )
    truth = qiime2.CategoricalMetadataColumn(truth)

    confusion = ctx.get_action('sample_classifier', 'confusion_matrix')
    accuracy_results, = confusion(
        predictions, truth, missing_samples='ignore', palette=palette
    )

    return predictions, accuracy_results


def classify_samples(ctx,
                     table,
                     metadata,
                     test_size=defaults['test_size'],
                     step=defaults['step'],
                     cv=defaults['cv'],
                     random_state=None,
                     n_jobs=defaults['n_jobs'],
                     n_estimators=defaults['n_estimators'],
                     estimator=defaults['estimator_c'],
                     optimize_feature_selection=False,
                     parameter_tuning=False,
                     palette=defaults['palette'],
                     missing_samples=defaults['missing_samples']):

    split = ctx.get_action('sample_classifier', 'split_table')
    fit = ctx.get_action('sample_classifier', 'fit_classifier')
    predict_test = ctx.get_action(
        'sample_classifier', 'predict_classification')
    summarize_estimator = ctx.get_action('sample_classifier', 'summarize')
    confusion = ctx.get_action('sample_classifier', 'confusion_matrix')
    heat = ctx.get_action('sample_classifier', 'heatmap')

    X_train, X_test, y_train, y_test = split(table, metadata, test_size,
                                             random_state,
                                             stratify=True,
                                             missing_samples=missing_samples)

    sample_estimator, importance = fit(
        X_train, metadata, step, cv, random_state, n_jobs, n_estimators,
        estimator, optimize_feature_selection, parameter_tuning,
        missing_samples='ignore')

    predictions, probabilities, = predict_test(
        X_test, sample_estimator, n_jobs)

    summary, = summarize_estimator(sample_estimator)

    accuracy_results, = confusion(predictions, metadata, probabilities,
                                  missing_samples='ignore', palette=palette)

    _heatmap, _ = heat(table, importance, sample_metadata=metadata,
                       group_samples=True, missing_samples=missing_samples)

    return (sample_estimator, importance, predictions, summary,
            accuracy_results, probabilities, _heatmap, y_train, y_test)


def regress_samples(ctx,
                    table,
                    metadata,
                    test_size=defaults['test_size'],
                    step=defaults['step'],
                    cv=defaults['cv'],
                    random_state=None,
                    n_jobs=defaults['n_jobs'],
                    n_estimators=defaults['n_estimators'],
                    estimator=defaults['estimator_r'],
                    optimize_feature_selection=False,
                    stratify=False,
                    parameter_tuning=False,
                    missing_samples=defaults['missing_samples']):

    split = ctx.get_action('sample_classifier', 'split_table')
    fit = ctx.get_action('sample_classifier', 'fit_regressor')
    predict_test = ctx.get_action('sample_classifier', 'predict_regression')
    summarize_estimator = ctx.get_action('sample_classifier', 'summarize')
    scatter = ctx.get_action('sample_classifier', 'scatterplot')

    X_train, X_test, y_train, y_test = split(table, metadata, test_size,
                                             random_state,
                                             stratify,
                                             missing_samples=missing_samples)

    sample_estimator, importance = fit(
        X_train, metadata, step, cv, random_state, n_jobs, n_estimators,
        estimator, optimize_feature_selection, parameter_tuning,
        missing_samples='ignore')

    predictions, = predict_test(X_test, sample_estimator, n_jobs)

    summary, = summarize_estimator(sample_estimator)

    accuracy_results, = scatter(predictions, metadata, 'ignore')

    return (sample_estimator, importance, predictions, summary,
            accuracy_results)


def fit_classifier(table: biom.Table,
                   metadata: qiime2.CategoricalMetadataColumn,
                   step: float = defaults['step'], cv: int = defaults['cv'],
                   random_state: int = None, n_jobs: int = defaults['n_jobs'],
                   n_estimators: int = defaults['n_estimators'],
                   estimator: str = defaults['estimator_c'],
                   optimize_feature_selection: bool = False,
                   parameter_tuning: bool = False,
                   missing_samples: str = defaults['missing_samples']
                   ) -> (Pipeline, pd.DataFrame):
    estimator, importance = _fit_estimator(
        table, metadata, estimator, n_estimators, step, cv, random_state,
        n_jobs, optimize_feature_selection, parameter_tuning,
        missing_samples=missing_samples, classification=True)

    return estimator, importance


def fit_regressor(table: biom.Table,
                  metadata: qiime2.CategoricalMetadataColumn,
                  step: float = defaults['step'], cv: int = defaults['cv'],
                  random_state: int = None, n_jobs: int = defaults['n_jobs'],
                  n_estimators: int = defaults['n_estimators'],
                  estimator: str = defaults['estimator_r'],
                  optimize_feature_selection: bool = False,
                  parameter_tuning: bool = False,
                  missing_samples: str = defaults['missing_samples']
                  ) -> (Pipeline, pd.DataFrame):
    estimator, importance = _fit_estimator(
        table, metadata, estimator, n_estimators, step, cv, random_state,
        n_jobs, optimize_feature_selection, parameter_tuning,
        missing_samples=missing_samples, classification=False)

    return estimator, importance


def predict_base(table, sample_estimator, n_jobs):
    if n_jobs == 0:
        n_jobs = get_available_cores()

    # extract feature data from biom
    feature_data = _extract_features(table)
    index = table.ids()

    # reset n_jobs if this is a valid parameter for the estimator
    if 'est__n_jobs' in sample_estimator.get_params().keys():
        sample_estimator.set_params(est__n_jobs=n_jobs)

    # predict values and output as series
    y_pred = sample_estimator.predict(feature_data)
    # need to flatten arrays that come out as multidimensional
    y_pred = y_pred.flatten()
    y_pred = pd.Series(y_pred, index=index, name='prediction')
    y_pred.index.name = 'SampleID'

    # log prediction probabilities (classifiers only)
    if sample_estimator.named_steps.est.__class__.__name__ in _classifiers:
        probs = predict_probabilities(sample_estimator, feature_data, index)
    else:
        probs = None

    return y_pred, probs


def predict_classification(table: biom.Table, sample_estimator: Pipeline,
                           n_jobs: int = defaults['n_jobs']) -> (
                            pd.Series, pd.DataFrame):
    return predict_base(table, sample_estimator, n_jobs)


def predict_regression(table: biom.Table, sample_estimator: Pipeline,
                       n_jobs: int = defaults['n_jobs']) -> pd.Series:
    # we only return the predictions, not the probabilities, which are empty
    # for regressors.
    return predict_base(table, sample_estimator, n_jobs)[0]


def split_table(table: biom.Table, metadata: qiime2.MetadataColumn,
                test_size: float = defaults['test_size'],
                random_state: int = None, stratify: str = True,
                missing_samples: str = defaults['missing_samples']
                ) -> (biom.Table, biom.Table, pd.Series, pd.Series):
    column = metadata.name
    X_train, X_test, y_train, y_test = _prepare_training_data(
        table, metadata, column, test_size, random_state, load_data=True,
        stratify=stratify, missing_samples=missing_samples)
    return X_train, X_test, y_train, y_test


def regress_samples_ncv(
        table: biom.Table, metadata: qiime2.NumericMetadataColumn,
        cv: int = defaults['cv'], random_state: int = None,
        n_jobs: int = defaults['n_jobs'],
        n_estimators: int = defaults['n_estimators'],
        estimator: str = defaults['estimator_r'], stratify: str = False,
        parameter_tuning: bool = False,
        missing_samples: str = defaults['missing_samples']
        ) -> (pd.Series, pd.DataFrame):

    y_pred, importances, probabilities = nested_cross_validation(
        table, metadata, cv, random_state, n_jobs, n_estimators, estimator,
        stratify, parameter_tuning, classification=False,
        scoring=mean_squared_error, missing_samples=missing_samples)
    return y_pred, importances


def classify_samples_ncv(
        table: biom.Table, metadata: qiime2.CategoricalMetadataColumn,
        cv: int = defaults['cv'], random_state: int = None,
        n_jobs: int = defaults['n_jobs'],
        n_estimators: int = defaults['n_estimators'],
        estimator: str = defaults['estimator_c'],
        parameter_tuning: bool = False,
        missing_samples: str = defaults['missing_samples']
        ) -> (pd.Series, pd.DataFrame, pd.DataFrame):

    y_pred, importances, probabilities = nested_cross_validation(
        table, metadata, cv, random_state, n_jobs, n_estimators, estimator,
        stratify=True, parameter_tuning=parameter_tuning, classification=False,
        scoring=accuracy_score, missing_samples=missing_samples)
    return y_pred, importances, probabilities


def scatterplot(output_dir: str, predictions: pd.Series,
                truth: qiime2.NumericMetadataColumn,
                missing_samples: str = defaults['missing_samples']) -> None:
    predictions = pd.to_numeric(predictions)

    _plot_accuracy(output_dir, predictions, truth, probabilities=None,
                   missing_samples=missing_samples,
                   classification=False, palette=None,
                   plot_title='regression scatterplot')


def confusion_matrix(output_dir: str,
                     predictions: pd.Series,
                     truth: qiime2.CategoricalMetadataColumn,
                     probabilities: pd.DataFrame = None,
                     missing_samples: str = defaults['missing_samples'],
                     vmin: int = 'auto', vmax: int = 'auto',
                     palette: str = defaults['palette']) -> None:

    if vmin == 'auto':
        vmin = None
    if vmax == 'auto':
        vmax = None

    predictions = predictions.astype(str)

    _plot_accuracy(output_dir, predictions, truth, probabilities,
                   missing_samples=missing_samples,
                   classification=True, palette=palette,
                   plot_title='confusion matrix', vmin=vmin, vmax=vmax)


def summarize(output_dir: str, sample_estimator: Pipeline):
    _summarize_estimator(output_dir, sample_estimator)


def heatmap(ctx, table, importance, sample_metadata=None,
            feature_metadata=None, feature_count=50,
            importance_threshold=0, group_samples=False, normalize=True,
            missing_samples='ignore', metric='braycurtis',
            method='average', cluster='features', color_scheme='rocket'):
    filter_features = ctx.get_action('feature_table', 'filter_features')
    group = ctx.get_action('feature_table', 'group')
    make_heatmap = ctx.get_action('feature_table', 'heatmap')
    filter_samples = ctx.get_action('feature_table', 'filter_samples')

    if group_samples and sample_metadata is None:
        raise ValueError(
            'If group_samples is enabled, sample_metadata are not optional.')

    if missing_samples == 'ignore' and sample_metadata is None:
        raise ValueError(
            'If missing_samples is ignore, metadata are not optional')

    clustermap_params = {
        'cluster': cluster, 'normalize': normalize, 'metric': metric,
        'method': method, 'color_scheme': color_scheme}

    # load importance data and sum rows (to average importances if there are
    # multiple scores).
    importance = importance.view(pd.DataFrame)
    importance = importance.sum(1)

    # filter importances by user criteria
    importance = importance.sort_values(ascending=False)
    if importance_threshold > 0:
        importance = importance[importance > importance_threshold]
    if feature_count > 0:
        importance = importance[:feature_count]
    importance.name = 'importance'
    importance = qiime2.Metadata(importance.to_frame())

    # filter features by importance
    table, = filter_features(table, metadata=importance)
    if missing_samples == 'ignore':
        table, = filter_samples(
            table, metadata=qiime2.Metadata(sample_metadata.to_dataframe()))

    # optionally group feature table by sample metadata
    # otherwise annotate heatmap with sample metadata
    if group_samples:
        table, = group(table, metadata=sample_metadata, axis='sample',
                       mode='sum')
    elif sample_metadata is not None:
        clustermap_params['sample_metadata'] = sample_metadata
    # label features using feature metadata
    if feature_metadata is not None:
        clustermap_params['feature_metadata'] = feature_metadata

    # make yer heatmap
    clustermap, = make_heatmap(table, **clustermap_params)

    return clustermap, table


# The following method is experimental and is not registered in the current
# release. Any use of the API is at user's own risk.
def detect_outliers(table: biom.Table,
                    metadata: qiime2.Metadata, subset_column: str = None,
                    subset_value: str = None,
                    n_estimators: int = defaults['n_estimators'],
                    contamination: float = 0.05, random_state: int = None,
                    n_jobs: int = defaults['n_jobs'],
                    missing_samples: str = 'ignore') -> (pd.Series):

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
    estimator = Pipeline([('dv', DictVectorizer()),
                          ('est', IsolationForest(n_jobs=n_jobs,
                                                  n_estimators=n_estimators,
                                                  contamination=contamination,
                                                  random_state=random_state,
                                                  ))])
    estimator.fit(X_train)

    # predict outlier status
    y_pred = estimator.predict(features)
    y_pred = pd.Series(y_pred, index=sample_md.index)
    # predict reports whether sample is an inlier; change to outlier status
    y_pred[y_pred == -1] = 'True'
    y_pred[y_pred == 1] = 'False'
    y_pred.name = "outlier"
    return y_pred
