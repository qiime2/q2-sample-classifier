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
                        _maz_score, _set_parameters_and_estimator,
                        _prepare_training_data, _disable_feature_selection,
                        nested_cross_validation, _fit_estimator,
                        _extract_features, _plot_accuracy,
                        _summarize_estimator, _validate_metadata_is_superset)


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

    X_train, X_test = split(table, metadata, test_size, random_state,
                            stratify=True, missing_samples=missing_samples)

    sample_estimator, importance = fit(
        X_train, metadata, step, cv, random_state, n_jobs, n_estimators,
        estimator, optimize_feature_selection, parameter_tuning,
        missing_samples='ignore')

    predictions, = predict_test(X_test, sample_estimator, n_jobs)

    summary, = summarize_estimator(sample_estimator)

    accuracy_results, = confusion(
        predictions, metadata, missing_samples='ignore', palette=palette)

    return sample_estimator, importance, predictions, summary, accuracy_results


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

    X_train, X_test = split(table, metadata, test_size, random_state,
                            stratify, missing_samples=missing_samples)

    sample_estimator, importance = fit(
        X_train, metadata, step, cv, random_state, n_jobs, n_estimators,
        estimator, optimize_feature_selection, parameter_tuning,
        missing_samples='ignore')

    predictions, = predict_test(X_test, sample_estimator, n_jobs)

    summary, = summarize_estimator(sample_estimator)

    accuracy_results, = scatter(predictions, metadata, 'ignore')

    return sample_estimator, importance, predictions, summary, accuracy_results


# this action has been replaced by the classify_samples pipeline and is no
# longer registered. Will be removed in a separate PR.
def classify_samples_basic(output_dir: str, table: biom.Table,
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
                           missing_samples: str=defaults['missing_samples']
                           ) -> None:

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


# this action has been replaced by the regress_samples pipeline and is no
# longer registered. Will be removed in a separate PR.
def regress_samples_basic(output_dir: str, table: biom.Table,
                          metadata: qiime2.NumericMetadataColumn,
                          test_size: float=defaults['test_size'],
                          step: float=defaults['step'],
                          cv: int=defaults['cv'], random_state: int=None,
                          n_jobs: int=defaults['n_jobs'],
                          n_estimators: int=defaults['n_estimators'],
                          estimator: str=defaults['estimator_r'],
                          optimize_feature_selection: bool=False,
                          stratify: str=False, parameter_tuning: bool=False,
                          missing_samples: str=defaults['missing_samples']
                          ) -> None:

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


def predict_base(table, sample_estimator, n_jobs):
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


def predict_classification(table: biom.Table, sample_estimator: Pipeline,
                           n_jobs: int=defaults['n_jobs']) -> pd.Series:
    return predict_base(table, sample_estimator, n_jobs)


def predict_regression(table: biom.Table, sample_estimator: Pipeline,
                       n_jobs: int=defaults['n_jobs']) -> pd.Series:
    return predict_base(table, sample_estimator, n_jobs)


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


def summarize(output_dir: str, sample_estimator: Pipeline):
    _summarize_estimator(output_dir, sample_estimator)


def maturity_index(ctx,
                   table,
                   metadata,
                   column,
                   group_by,
                   control,
                   estimator=defaults['estimator_r'],
                   n_estimators=defaults['n_estimators'],
                   test_size=0.5,
                   step=defaults['step'],
                   cv=defaults['cv'],
                   random_state=None,
                   n_jobs=defaults['n_jobs'],
                   parameter_tuning=False,
                   optimize_feature_selection=False,
                   stratify=False,
                   missing_samples=defaults['missing_samples']):

    filter_samples = ctx.get_action('feature_table', 'filter_samples')
    filter_features = ctx.get_action('feature_table', 'filter_features')
    group_table = ctx.get_action('feature_table', 'group')
    heatmap = ctx.get_action('feature_table', 'heatmap')
    split = ctx.get_action('sample_classifier', 'split_table')
    fit = ctx.get_action('sample_classifier', 'fit_regressor')
    predict_test = ctx.get_action('sample_classifier', 'predict')
    summarize_estimator = ctx.get_action('sample_classifier', 'summarize')
    scatter = ctx.get_action('sample_classifier', 'scatterplot')
    volatility = ctx.get_action('longitudinal', 'volatility')

    # we must perform metadata superset validation here before we start
    # slicing and dicing.
    md_as_frame = metadata.to_dataframe()
    if missing_samples == 'error':
        _validate_metadata_is_superset(md_as_frame, table.view(biom.Table))

    # train regressor on subset of control samples
    control_table, = filter_samples(
        table, metadata=metadata, where="{0}='{1}'".format(group_by, control))

    md_column = metadata.get_column(column)
    X_train, X_test = split(control_table, md_column, test_size, random_state,
                            stratify, missing_samples='ignore')

    sample_estimator, importance = fit(
        X_train, md_column, step, cv, random_state, n_jobs, n_estimators,
        estimator, optimize_feature_selection, parameter_tuning,
        missing_samples='ignore')

    # drop training samples from rest of dataset; we will predict all others
    control_ids = pd.DataFrame(index=X_train.view(biom.Table).ids())
    control_ids.index.name = 'id'
    control_ids = qiime2.Metadata(control_ids)
    test_table, = filter_samples(table, metadata=control_ids, exclude_ids=True)

    # predict test samples
    predictions, = predict_test(test_table, sample_estimator, n_jobs)

    # summarize estimator params
    summary, = summarize_estimator(sample_estimator)

    # only report accuracy on control test samples
    test_ids = X_test.view(biom.Table).ids()
    accuracy_md = metadata.filter_ids(test_ids).get_column(column)
    accuracy_results, = scatter(predictions, accuracy_md, 'ignore')

    # calculate MAZ score
    # merge is inner join by default, so training samples are dropped (good!)
    pred_md = metadata.merge(predictions.view(qiime2.Metadata)).to_dataframe()
    pred_md['prediction'] = pd.to_numeric(pred_md['prediction'])
    pred_md = _maz_score(pred_md, 'prediction', column, group_by, control)
    maz = '{0} MAZ score'.format(column)
    maz_scores = qiime2.Artifact.import_data(
        'SampleData[Predictions]', pred_md[maz])

    # make heatmap
    # trim table to important features for viewing as heatmap
    table, = filter_features(table, metadata=importance.view(qiime2.Metadata))
    # make sure IDs match between table and metadata
    cluster_table, = filter_samples(table, metadata=metadata)
    # need to group table by two columns together, so do this ugly hack
    cluster_by = group_by + '-' + column
    md_as_frame[cluster_by] = (md_as_frame[group_by].astype(str) + '-' +
                               md_as_frame[column].astype(str))
    cluster_md = qiime2.CategoricalMetadataColumn(md_as_frame[cluster_by])
    cluster_table, = group_table(cluster_table, axis='sample',
                                 metadata=cluster_md, mode='median-ceiling')
    # group metadata to match grouped sample IDs and sort by group/column
    clust_md = md_as_frame.groupby(cluster_by).first()
    clust_md = clust_md.sort_values([group_by, column])
    # sort table using clustered/sorted metadata as guide
    sorted_table = cluster_table.view(biom.Table).sort_order(clust_md.index)
    sorted_table = qiime2.Artifact.import_data(
        'FeatureTable[Frequency]', sorted_table)
    clustermap, = heatmap(sorted_table, cluster='features')

    # visualize MAZ vs. time (column)
    lineplots, = volatility(
        qiime2.Metadata(pred_md), state_column=column,
        individual_id_column=None, default_group_column=group_by,
        default_metric=maz, yscale='linear')

    return (
        sample_estimator, importance, predictions, summary, accuracy_results,
        maz_scores, clustermap, lineplots)


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
