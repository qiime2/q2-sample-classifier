# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import warnings
from os.path import join

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, KFold, StratifiedKFold)
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline

import q2templates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pkg_resources
from scipy.sparse import issparse
from scipy.stats import randint
import biom

from .visuals import (_linear_regress, _plot_confusion_matrix, _plot_RFE,
                      _regplot_from_dataframe, _generate_roc_plots)

_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier',
                'GradientBoostingClassifier', 'AdaBoostClassifier',
                'KNeighborsClassifier', 'LinearSVC', 'SVC']

parameters = {
    'ensemble': {"max_depth": [4, 8, 16, None],
                 "max_features": [None, 'sqrt', 'log2', 0.1],
                 "min_samples_split": [0.001, 0.01, 0.1],
                 "min_weight_fraction_leaf": [0.0001, 0.001, 0.01]},
    'bootstrap': {"bootstrap": [True, False]},
    'criterion': {"criterion": ["gini", "entropy"]},
    'svm': {"C": [1, 0.5, 0.1, 0.9, 0.8],
            "tol": [0.00001, 0.0001, 0.001, 0.01],
            "shrinking": [True, False]},
    'kneighbors': {"n_neighbors": randint(2, 15),
                   "weights": ['uniform', 'distance'],
                   "leaf_size": randint(15, 100)},
    'linear': {"alpha": [0.0001, 0.01, 1.0, 10.0, 1000.0],
               "tol": [0.00001, 0.0001, 0.001, 0.01]}
}


TEMPLATES = pkg_resources.resource_filename('q2_sample_classifier', 'assets')


def _extract_features(feature_data):
    ids = feature_data.ids('observation')
    features = np.empty(feature_data.shape[1], dtype=dict)
    for i, row in enumerate(feature_data.matrix_data.T):
        features[i] = {ids[ix]: d for ix, d in zip(row.indices, row.data)}
    return features


def _load_data(feature_data, targets_metadata, missing_samples, extract=True):
    '''Load data and generate training and test sets.

    feature_data: pd.DataFrame
        feature X sample values.
    targets_metadata: qiime2.Metadata
        target (columns) X sample (rows) values.
    '''
    # Load metadata, attempt to convert to numeric
    targets = targets_metadata.to_dataframe()

    if missing_samples == 'error':
        _validate_metadata_is_superset(targets, feature_data)

    # filter features and targest so samples match
    index = set(targets.index)
    index = [ix for ix in feature_data.ids() if ix in index]
    targets = targets.loc[index]
    feature_data = feature_data.filter(index, inplace=False)
    if extract:
        feature_data = _extract_features(feature_data)

    return feature_data, targets


def _validate_metadata_is_superset(metadata, table):
    metadata_ids = set(metadata.index.tolist())
    table_ids = set(table.ids())
    missing_ids = table_ids.difference(metadata_ids)
    if len(missing_ids) > 0:
        raise ValueError('Missing samples in metadata: %r' % missing_ids)


def _extract_important_features(index, top):
    '''Find top features, match names to indices, sort.
    index: ndarray
        Feature names
    top: array
        Feature importance scores, coef_ scores, or ranking of scores.
    '''
    # is top a 1-d or multi-d array?
    # coef_ is a multidimensional array of shape = [n_class-1, n_features]
    if any(isinstance(i, list) for i in top) or top.ndim > 1:
        if issparse(top):
            top = top.todense()
        imp = pd.DataFrame(
            top, index=["importance{0}".format(n) for n in range(len(top))]).T
    # ensemble estimators and RFECV return 1-d arrays
    else:
        imp = pd.DataFrame(top, columns=["importance"])
    imp.index = index
    imp.index.name = 'feature'
    imp = sort_importances(imp, ascending=False)
    return imp


def _split_training_data(feature_data, targets, column, test_size=0.2,
                         stratify=None, random_state=None, drop_na=True):
    '''Split data sets into training and test sets.

    feature_data: biom.Table
        feature X sample values.
    targets: pandas.DataFrame
        target (columns) X sample (rows) values.
    column: str
        Target column contained in targets.
    test_size: float
        Fraction of data to be reserved as test data.
    stratify: array-like
        Stratify data using this as class labels. E.g., set to df
        column by setting stratify=df[column]
    random_state: int or None
        Int to use for seeding random state. Random if None.
    '''
    # Define target / predictor data
    targets = targets[column]

    if drop_na:
        targets = targets.dropna()

    if test_size > 0.0:
        try:
            y_train, y_test = train_test_split(
                targets, test_size=test_size, stratify=stratify,
                random_state=random_state)
        except ValueError:
            _stratification_error()
    else:
        warning_msg = _warn_zero_test_split()
        warnings.warn(warning_msg, UserWarning)

        X_train, X_test, y_train, y_test = (
            feature_data, feature_data, targets, targets)

    tri = y_train.index
    # filter and sort biom tables to match split/filtered metadata ids
    # skip filtering if no splitting/dropna was performed
    # if test_size > 0.0 is implicit, so don't need to worry about initializing
    # X_train and X_test in an else statement.
    if list(tri) != list(feature_data.ids()):
        tei = y_test.index
        X_train = feature_data.filter(tri, inplace=False).sort_order(tri)
        X_test = feature_data.filter(tei, inplace=False).sort_order(tei)

    return X_train, X_test, y_train, y_test


def _stratification_error():
    raise ValueError((
        'You have chosen to predict a metadata column that contains '
        'one or more values that match only one sample. For proper '
        'stratification of data into training and test sets, each '
        'class (value) must contain at least two samples. This is a '
        'requirement for classification problems, but stratification '
        'can be disabled for regression by setting stratify=False. '
        'Alternatively, remove all samples that bear a unique class '
        'label for your chosen metadata column. Note that disabling '
        'stratification can negatively impact predictive accuracy for '
        'small data sets.'))


def _rfecv_feature_selection(feature_data, targets, estimator,
                             cv=5, step=1, scoring=None, n_jobs=1):
    '''Optimize feature depth by testing model accuracy at
    multiple feature depths with cross-validated recursive
    feature elimination.
    __________
    Parameters
    __________
    feature_data: list of dicts
        Training set feature data x samples.
    targets: pandas.DataFrame
        Training set target value data x samples.
    cv: int
        Number of k-fold cross-validations to perform.
    step: float or int
        If float, reduce this fraction of features at each step.
        If int, reduce this number of features at each step.
    estimator: sklearn classifier
        estimator to use, with parameters set. If none, default
        to random forests.
    n_jobs: int
        Number of parallel jobs to run.

    For other params, see sklearn.ensemble.RandomForestRegressor.

    __________
    Returns
    __________
    rfecv: sklearn estimator
        Can be used to predict target values for test data.
    importance: pandas.DataFrame
        List of top features.
    '''

    rfecv = Pipeline(
        [('dv', estimator.named_steps.dv),
         ('est', RFECV(estimator=estimator.named_steps.est, step=step, cv=cv,
                       scoring=scoring, n_jobs=n_jobs))])

    rfecv.fit(feature_data, targets.values.ravel())

    # Describe top features
    n_opt = rfecv.named_steps.est.n_features_
    importance = _extract_important_features(
        rfecv.named_steps.dv.get_feature_names(),
        rfecv.named_steps.est.ranking_)
    importance = sort_importances(importance, ascending=True)[:n_opt]

    rfe_scores = _extract_rfe_scores(rfecv.named_steps.est)

    return importance, rfe_scores


def _extract_rfe_scores(rfecv):
    n_features = len(rfecv.ranking_)
    # If using fractional step, step = integer of fraction * n_features
    if rfecv.step < 1:
        rfecv.step = int(rfecv.step * n_features)
    # Need to manually calculate x-axis, as rfecv.grid_scores_ are a 1-d array
    x = [n_features - (n * rfecv.step)
         for n in range(len(rfecv.grid_scores_)-1, -1, -1)]
    if x[0] < 1:
        x[0] = 1
    return pd.Series(rfecv.grid_scores_, index=x, name='Accuracy')


def nested_cross_validation(table, metadata, cv, random_state, n_jobs,
                            n_estimators, estimator, stratify,
                            parameter_tuning, classification, scoring,
                            missing_samples='error'):
    # extract column name from NumericMetadataColumn
    column = metadata.name

    # load feature data, metadata targets
    X_train, y_train = _load_data(
        table, metadata, missing_samples=missing_samples)

    # disable feature selection for unsupported estimators
    optimize_feature_selection, calc_feature_importance = \
        _disable_feature_selection(estimator, False)

    # specify parameters and distributions to sample from for parameter tuning
    estimator, param_dist, parameter_tuning = _set_parameters_and_estimator(
        estimator, table, y_train[column], column, n_estimators, n_jobs, cv,
        random_state, parameter_tuning, classification)

    # predict values for all samples via (nested) CV
    scores, predictions, importances, tops, probabilities = \
        _fit_and_predict_cv(
            X_train, y_train[column], estimator, param_dist, n_jobs, scoring,
            random_state, cv, stratify, calc_feature_importance,
            parameter_tuning)

    # Print accuracy score to stdout
    print("Estimator Accuracy: {0} ± {1}".format(
        np.mean(scores), np.std(scores)))

    # TODO: save down estimator with tops parameters (currently the estimator
    # would be untrained, and tops parameters are not reported)

    return predictions['prediction'], importances, probabilities


def _fit_estimator(features, targets, estimator, n_estimators=100, step=0.05,
                   cv=5, random_state=None, n_jobs=1,
                   optimize_feature_selection=False, parameter_tuning=False,
                   missing_samples='error', classification=True):
    # extract column name from CategoricalMetadataColumn
    column = targets.to_series().name

    # load data
    X_train, y_train = _load_data(
        features, targets, missing_samples=missing_samples)

    # disable feature selection for unsupported estimators
    optimize_feature_selection, calc_feature_importance = \
        _disable_feature_selection(estimator, optimize_feature_selection)

    # specify parameters and distributions to sample from for parameter tuning
    estimator, param_dist, parameter_tuning = _set_parameters_and_estimator(
        estimator, features, targets, column, n_estimators, n_jobs, cv,
        random_state, parameter_tuning, classification=classification)

    # optimize training feature count
    if optimize_feature_selection:
        X_train, importances, rfe_scores = _optimize_feature_selection(
            X_train=X_train, y_train=y_train,
            estimator=estimator, cv=cv, step=step, n_jobs=n_jobs)
    else:
        importances = None

    # optimize tuning parameters on your training set
    if parameter_tuning:
        # tune parameters
        estimator = _tune_parameters(
            X_train, y_train, estimator, param_dist, n_iter_search=20,
            n_jobs=n_jobs, cv=cv, random_state=random_state).best_estimator_

    # fit estimator
    estimator.fit(X_train, y_train.values.ravel())

    importances = _attempt_to_calculate_feature_importances(
        estimator, calc_feature_importance,
        optimize_feature_selection, importances)

    if optimize_feature_selection:
        estimator.rfe_scores = rfe_scores

    # TODO: drop this when we get around to supporting optional outputs
    # methods cannot output an empty importances artifact; only KNN has no
    # feature importance, but just warn and output all features as
    # importance = 1
    if importances is None:
        _warn_feature_selection()
        importances = pd.DataFrame(index=features.ids('observation'))
        importances["importance"] = np.nan
        importances.index.name = 'feature'

    return estimator, importances


def _attempt_to_calculate_feature_importances(
        estimator, calc_feature_importance,
        optimize_feature_selection, importances=None):
    # calculate feature importances, if appropriate for the estimator
    if calc_feature_importance:
        importances = _calculate_feature_importances(estimator)
    # otherwise, if optimizing feature selection, just return ranking from RFE
    elif optimize_feature_selection:
        pass
    # otherwise, we have no weights nor selection, so features==n_features
    else:
        importances = None
    return importances


def _prepare_training_data(features, targets, column, test_size,
                           random_state, load_data=True, stratify=True,
                           missing_samples='error'):
    # load data
    if load_data:
        features, targets = _load_data(
            features, targets, missing_samples=missing_samples, extract=False)

    # split into training and test sets
    if stratify:
        strata = targets[column]
    else:
        strata = None

    X_train, X_test, y_train, y_test = _split_training_data(
        features, targets, column, test_size, strata, random_state)

    return X_train, X_test, y_train, y_test


def _optimize_feature_selection(X_train, y_train, estimator, cv, step, n_jobs):
    importance, rfe_scores = _rfecv_feature_selection(
        X_train, y_train, estimator=estimator, cv=cv, step=step, n_jobs=n_jobs)

    index = set(importance.index)
    X_train = [{k: r[k] for k in r.keys() & index} for r in X_train]
    return X_train, importance, rfe_scores


def _calculate_feature_importances(estimator):
    # only set calc_feature_importance=True if estimator has attributes
    # feature_importances_ or coef_ to report feature importance/weights
    try:
        importances = _extract_important_features(
            estimator.named_steps.dv.get_feature_names(),
            estimator.named_steps.est.feature_importances_)
    # is there a better way to determine whether estimator has coef_ ?
    except AttributeError:
        importances = _extract_important_features(
            estimator.named_steps.dv.get_feature_names(),
            estimator.named_steps.est.coef_)
    return importances


def _predict_and_plot(output_dir, y_test, y_pred, vmin=None, vmax=None,
                      classification=True, palette='sirocco'):
    if classification:
        x_classes = set(y_test.unique())
        y_classes = set(y_pred.unique())
        # validate: if classes are exclusive, accuracy is zero; user probably
        # input the wrong data!
        if len(x_classes.intersection(y_classes)) < 1:
            raise _class_overlap_error()
        else:
            classes = sorted(list(x_classes.union(y_classes)))
        predictions, predict_plot = _plot_confusion_matrix(
            y_test, y_pred, classes, normalize=True, palette=palette,
            vmin=vmin, vmax=vmax)
    else:
        predictions = _linear_regress(y_test, y_pred)
        predict_plot = _regplot_from_dataframe(y_test, y_pred)

    if output_dir is not None:
        predict_plot.get_figure().savefig(
            join(output_dir, 'predictions.png'), bbox_inches='tight')
        predict_plot.get_figure().savefig(
            join(output_dir, 'predictions.pdf'), bbox_inches='tight')

    plt.close('all')
    return predictions, predict_plot


def _class_overlap_error():
    raise ValueError(
        'Predicted and true metadata values do not overlap. Check your '
        'inputs to ensure that you are using the correct data. Is the '
        'correct metadata column being compared to these predictions? Was '
        'your model trained on the correct type of data? Prediction '
        'sample classes (metadata values) should match or be a subset of '
        'training sample classes. If you are attempting to calculate '
        'accuracy scores on predictions from a sample regressor, use '
        'scatterplot instead.')


def _match_series_or_die(predictions, truth, missing_samples='error'):
    # validate input metadata and predictions, output intersection.
    # truth must be a superset of predictions
    truth_ids = set(truth.index)
    predictions_ids = set(predictions.index)
    missing_ids = predictions_ids - truth_ids
    if missing_samples == 'error' and len(missing_ids) > 0:
        raise ValueError('Missing samples in metadata: %r' % missing_ids)

    # match metadata / prediction IDs
    predictions, truth = predictions.align(truth, axis=0, join='inner')

    return predictions, truth


def _plot_accuracy(output_dir, predictions, truth, probabilities,
                   missing_samples, classification, palette, plot_title,
                   vmin=None, vmax=None):
    '''Plot accuracy results and send to visualizer on either categorical
    or numeric data inside two pd.Series
    '''
    truth = truth.to_series()

    # check if test_size == 0.0 and all predictions are complete dataset
    if (missing_samples == 'ignore') & (
            predictions.shape[0] == truth.shape[0]):
        warning_msg = _warn_zero_test_split()
    else:
        warning_msg = None

    predictions, truth = _match_series_or_die(
        predictions, truth, missing_samples)

    # calculate prediction accuracy and plot results
    predictions, predict_plot = _predict_and_plot(
        output_dir, truth, predictions, vmin=vmin, vmax=vmax,
        classification=classification, palette=palette)

    # optionally generate ROC curves for classification results
    if probabilities is not None:
        probabilities, truth = _match_series_or_die(
            probabilities, truth, missing_samples)
        roc = _generate_roc_plots(truth, probabilities, palette)
        roc.savefig(join(output_dir, 'roc_plot.png'), bbox_inches='tight')
        roc.savefig(join(output_dir, 'roc_plot.pdf'), bbox_inches='tight')

    # output to viz
    _visualize(output_dir=output_dir, estimator=None, cm=predictions,
               roc=probabilities, optimize_feature_selection=False,
               title=plot_title, warning_msg=warning_msg)


def sort_importances(importances, ascending=False):
    return importances.sort_values(
        by=importances.columns[0], ascending=ascending)


def _extract_estimator_parameters(estimator):
    # summarize model accuracy and params
    # (drop pipeline params and individual base estimators)
    estimator_params = {k: v for k, v in estimator.get_params().items() if
                        k.startswith('est__') and k != 'est__base_estimator'}
    return pd.Series(estimator_params, name='Parameter setting')


def _summarize_estimator(output_dir, sample_estimator):
    try:
        rfep = _plot_RFE(
            x=sample_estimator.rfe_scores.index, y=sample_estimator.rfe_scores)
        rfep.savefig(join(output_dir, 'rfe_plot.png'))
        rfep.savefig(join(output_dir, 'rfe_plot.pdf'))
        plt.close('all')
        optimize_feature_selection = True
        # generate rfe scores file
        df = pd.DataFrame(data={'rfe_score': sample_estimator.rfe_scores},
                          index=sample_estimator.rfe_scores.index)
        df.index.name = 'feature_count'
        df.to_csv(join(output_dir, 'rfe_scores.tsv'), sep='\t', index=True)
    # if the rfe_scores attribute does not exist, do nothing
    except AttributeError:
        optimize_feature_selection = False

    _visualize(output_dir=output_dir, estimator=sample_estimator, cm=None,
               roc=None, optimize_feature_selection=optimize_feature_selection,
               title='Estimator Summary')


def _visualize(output_dir, estimator, cm, roc,
               optimize_feature_selection=True, title='results',
               warning_msg=None):

    pd.set_option('display.max_colwidth', None)

    # summarize model accuracy and params
    if estimator is not None:
        result = _extract_estimator_parameters(estimator)
        result = q2templates.df_to_html(result.to_frame())
    else:
        result = False

    if cm is not None:
        cm.to_csv(join(
            output_dir, 'predictive_accuracy.tsv'), sep='\t', index=True)
        cm = q2templates.df_to_html(cm)

    if roc is not None:
        roc = True

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': title,
        'result': result,
        'predictions': cm,
        'roc': roc,
        'optimize_feature_selection': optimize_feature_selection,
        'warning_msg': warning_msg})


def _visualize_knn(output_dir, params: pd.Series):
    result = q2templates.df_to_html(params.to_frame())
    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': 'Estimator Summary',
        'result': result,
        'predictions': None,
        'importances': None,
        'classification': True,
        'optimize_feature_selection': False})


def _map_params_to_pipeline(param_dist):
    return {'est__' + param: dist for param, dist in param_dist.items()}


def _tune_parameters(X_train, y_train, estimator, param_dist, n_iter_search=20,
                     n_jobs=1, cv=None, random_state=None):
    # run randomized search
    random_search = RandomizedSearchCV(
        estimator, param_distributions=param_dist, n_iter=n_iter_search,
        n_jobs=n_jobs, cv=cv, random_state=random_state)
    random_search.fit(X_train, y_train.values.ravel())
    return random_search


def _fit_and_predict_cv(table, metadata, estimator, param_dist, n_jobs,
                        scoring=accuracy_score, random_state=None, cv=10,
                        stratify=True, calc_feature_importance=False,
                        parameter_tuning=False):
    '''train and test estimators via cross-validation.
    scoring: str
        use accuracy_score for classification, mean_squared_error for
        regression.
    '''
    # Set CV method
    if stratify:
        _cv = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state)
    else:
        _cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    predictions = pd.DataFrame()
    probabilities = pd.DataFrame()
    scores = []
    top_params = []
    importances = []
    if isinstance(table, biom.Table):
        features = _extract_features(table)
    else:
        features = table
    for train_index, test_index in _cv.split(features, metadata):
        X_train = features[train_index]
        y_train = metadata.iloc[train_index]
        # perform parameter tuning in inner loop
        if parameter_tuning:
            estimator = _tune_parameters(
                X_train, y_train, estimator, param_dist,
                n_iter_search=20, n_jobs=n_jobs, cv=cv,
                random_state=random_state).best_estimator_
        else:
            # fit estimator on inner outer training set
            estimator.fit(X_train, y_train.values.ravel())
        # predict values for outer loop test set
        test_set = features[test_index]
        index = metadata.iloc[test_index]
        pred = pd.DataFrame(estimator.predict(test_set), index=index.index)

        # log predictions results
        predictions = pd.concat([predictions, pred])

        # log prediction probabilities (classifiers only)
        if estimator.named_steps.est.__class__.__name__ in _classifiers:
            probs = predict_probabilities(estimator, test_set, index.index)
            probabilities = pd.concat([probabilities, probs])

        # log accuracy on that fold
        scores += [scoring(pred, index)]
        # log feature importances
        if calc_feature_importance:
            imp = _calculate_feature_importances(estimator)
            importances += [imp]
        # log top parameters
        # for now we will cast as a str (instead of dict) so that we can count
        # frequency of unique elements below
        top_params += [str(estimator.named_steps.est.get_params())]

    # Report most frequent best params
    # convert top_params to a set, order by count (hence str conversion above)
    # max will be the most frequent... then we convert back to a dict via eval
    # which should be safe since this is always a dict of param values reported
    # by sklearn.
    tops = max(set(top_params), key=top_params.count)
    tops = eval(tops)

    # calculate mean feature importances
    if calc_feature_importance:
        importances = _mean_feature_importance(importances)
    else:
        importances = _null_feature_importance(table)

    predictions.columns = ['prediction']
    predictions.index.name = 'SampleID'
    probabilities.index.name = 'SampleID'

    return scores, predictions, importances, tops, probabilities


def predict_probabilities(estimator, test_set, index):
    '''
    Predict class probabilities for a set of test samples.

    estimator: sklearn trained classifier
    test_set: array-like of y_values (features) for test set samples that will
              have their class probabilities predicted.
    index: array-like of sample names
    '''
    # all used classifiers have a predict_proba attribute
    # (approximated for SVCs)
    probs = pd.DataFrame(estimator.predict_proba(test_set),
                         index=index, columns=estimator.classes_)

    return probs


def _mean_feature_importance(importances):
    '''Calculate mean feature importance across a list of pd.dataframes
    containing importance scores of the same features from multiple models
    (e.g., CV importance scores).
    '''
    imp = pd.concat(importances, axis=1, sort=True)
    # groupby column name instead of taking column mean to support 2d arrays
    imp = imp.groupby(imp.columns, axis=1).mean()
    return imp.sort_values(imp.columns[0], ascending=False)


def _null_feature_importance(table):
    feature_extractor = DictVectorizer()
    feature_extractor.fit(table)
    imp = pd.DataFrame(index=feature_extractor.get_feature_names())
    imp.index.name = "feature"
    imp["importance"] = 1
    return imp


def _select_estimator(estimator, n_jobs, n_estimators, random_state=None):
    '''Select estimator and parameters from argument name.'''
    # Regressors
    if estimator == 'RandomForestRegressor':
        param_dist = {**parameters['ensemble'], **parameters['bootstrap']}
        estimator = RandomForestRegressor(
            n_jobs=n_jobs, n_estimators=n_estimators,
            random_state=random_state)
    elif estimator == 'ExtraTreesRegressor':
        param_dist = {**parameters['ensemble'], **parameters['bootstrap']}
        estimator = ExtraTreesRegressor(
            n_jobs=n_jobs, n_estimators=n_estimators,
            random_state=random_state)
    elif estimator == 'GradientBoostingRegressor':
        param_dist = parameters['ensemble']
        estimator = GradientBoostingRegressor(
            n_estimators=n_estimators, random_state=random_state)
    elif estimator == 'SVR':
        param_dist = {**parameters['svm'], 'epsilon': [0.0, 0.1]}
        estimator = SVR(kernel='rbf', gamma='scale')
    elif estimator == 'LinearSVR':
        param_dist = {**parameters['svm'], 'epsilon': [0.0, 0.1]}
        estimator = SVR(kernel='linear')
    elif estimator == 'Ridge':
        param_dist = parameters['linear']
        estimator = Ridge(solver='auto', random_state=random_state)
    elif estimator == 'Lasso':
        param_dist = parameters['linear']
        estimator = Lasso(random_state=random_state)
    elif estimator == 'ElasticNet':
        param_dist = parameters['linear']
        estimator = ElasticNet(random_state=random_state)
    elif estimator == 'KNeighborsRegressor':
        param_dist = parameters['kneighbors']
        estimator = KNeighborsRegressor(algorithm='auto')

    # Classifiers
    elif estimator == 'RandomForestClassifier':
        param_dist = {**parameters['ensemble'], **parameters['bootstrap'],
                      **parameters['criterion']}
        estimator = RandomForestClassifier(
            n_jobs=n_jobs, n_estimators=n_estimators,
            random_state=random_state)
    elif estimator == 'ExtraTreesClassifier':
        param_dist = {**parameters['ensemble'], **parameters['bootstrap'],
                      **parameters['criterion']}
        estimator = ExtraTreesClassifier(
            n_jobs=n_jobs, n_estimators=n_estimators,
            random_state=random_state)
    elif estimator == 'GradientBoostingClassifier':
        param_dist = parameters['ensemble']
        estimator = GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state)
    elif estimator == 'LinearSVC':
        param_dist = parameters['svm']
        estimator = SVC(kernel='linear', random_state=random_state,
                        gamma='scale', probability=True)
    elif estimator == 'SVC':
        param_dist = parameters['svm']
        estimator = SVC(kernel='rbf', random_state=random_state,
                        gamma='scale', probability=True)
    elif estimator == 'KNeighborsClassifier':
        param_dist = parameters['kneighbors']
        estimator = KNeighborsClassifier(algorithm='auto')

    return param_dist, estimator


def _train_adaboost_base_estimator(table, metadata, column, n_estimators,
                                   n_jobs, cv, random_state=None,
                                   parameter_tuning=False,
                                   classification=True,
                                   missing_samples='error'):
    param_dist = parameters['ensemble']
    if classification:
        base_estimator = DecisionTreeClassifier()
        adaboost_estimator = AdaBoostClassifier
    else:
        base_estimator = DecisionTreeRegressor()
        adaboost_estimator = AdaBoostRegressor
    base_estimator = Pipeline(
        [('dv', DictVectorizer()), ('est', base_estimator)])

    if parameter_tuning:
        features, targets = _load_data(
            table, metadata, missing_samples=missing_samples)
        param_dist = _map_params_to_pipeline(param_dist)
        base_estimator = _tune_parameters(
            features, targets[column], base_estimator, param_dist,
            n_jobs=n_jobs, cv=cv, random_state=random_state).best_estimator_

    return Pipeline(
        [('dv', base_estimator.named_steps.dv),
         ('est', adaboost_estimator(base_estimator.named_steps.est,
                                    n_estimators, random_state=random_state))])


def _disable_feature_selection(estimator, optimize_feature_selection):
    '''disable feature selection for unsupported classifiers.'''

    unsupported = ['KNeighborsClassifier', 'SVC', 'KNeighborsRegressor', 'SVR']

    if estimator in unsupported:
        optimize_feature_selection = False
        calc_feature_importance = False
        _warn_feature_selection()
    else:
        calc_feature_importance = True

    return optimize_feature_selection, calc_feature_importance


def _set_parameters_and_estimator(estimator, table, metadata, column,
                                  n_estimators, n_jobs, cv, random_state,
                                  parameter_tuning, classification=True,
                                  missing_samples='error'):
    # specify parameters and distributions to sample from for parameter tuning
    if estimator in ['AdaBoostClassifier', 'AdaBoostRegressor']:
        estimator = _train_adaboost_base_estimator(
            table, metadata, column, n_estimators, n_jobs, cv, random_state,
            parameter_tuning, classification=classification,
            missing_samples=missing_samples)
        parameter_tuning = False
        param_dist = None
    else:
        param_dist, estimator = _select_estimator(
            estimator, n_jobs, n_estimators, random_state)
        estimator = Pipeline([('dv', DictVectorizer()), ('est', estimator)])
        param_dist = _map_params_to_pipeline(param_dist)
    return estimator, param_dist, parameter_tuning


def _warn_feature_selection():
    warning = (
        ('This estimator does not support recursive feature extraction with '
         'the parameter settings requested. See documentation or try a '
         'different estimator model.'))
    warnings.warn(warning, UserWarning)


def _warn_zero_test_split():
    return 'Using test_size = 0.0, you are using your complete dataset for ' \
        'fitting the estimator. Hence, any returned model evaluations are ' \
        'based on that same training dataset and are not representative of ' \
        'your model\'s performance on a previously unseen dataset. Please ' \
        'consider evaluating this model on a separate dataset.'
