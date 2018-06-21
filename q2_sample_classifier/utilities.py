# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
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
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline

import q2templates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pkg_resources
from scipy.stats import randint
import biom

from .visuals import (_linear_regress, _plot_confusion_matrix, _plot_RFE,
                      _pairwise_stats, _two_way_anova, _regplot_from_dataframe,
                      _boxplot_from_dataframe, _lmplot_from_dataframe,
                      _clustermap_from_dataframe)


parameters = {
    'ensemble': {"max_depth": [4, 8, 16, None],
                 "max_features": [None, 'sqrt', 'log2', 0.1],
                 "min_samples_split": [0.001, 0.01, 0.1],
                 "min_weight_fraction_leaf": [0.0001, 0.001, 0.01]},
    'bootstrap': {"bootstrap": [True, False]},
    'criterion': {"criterion": ["gini", "entropy"]},
    'linear_svm': {"C": [1, 0.5, 0.1, 0.9, 0.8],
                   # should probably include penalty in grid search, but:
                   # Unsupported set of arguments: The combination of
                   # penalty='l1' and loss='hinge' is not supported
                   # "penalty": ["l1", "l2"],
                   "loss": ["hinge", "squared_hinge"],
                   "tol": [0.00001, 0.0001, 0.001]
                   # should probably include this in grid search, as
                   # dual=False is preferred when samples>features. However:
                   # Unsupported set of arguments: The combination of
                   # penalty='l2' and loss='hinge' are not supported when
                   # dual=False
                   # "dual": [True, False]
                   },
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


def _load_data(feature_data, targets_metadata, missing_samples):
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

    feature_data: pandas.DataFrame
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
        try:
            targets, feature_data = \
                zip(*[(t, f) for t, f in zip(targets, feature_data)
                      if pd.notna(t)])
        except ValueError:
            targets, feature_data = [], []
        targets = pd.Series(targets)
        targets.name = column

    if test_size > 0.0:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                feature_data, targets, test_size=test_size, stratify=stratify,
                random_state=random_state)
        except ValueError:
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
    else:
        X_train, X_test, y_train, y_test = (
            feature_data, feature_data, targets, targets)

    return X_train, X_test, y_train, y_test


def _rfecv_feature_selection(feature_data, targets, estimator,
                             cv=5, step=1, scoring=None, n_jobs=1):
    '''Optimize feature depth by testing model accuracy at
    multiple feature depths with cross-validated recursive
    feature elimination.
    __________
    Parameters
    __________
    feature_data: pandas.DataFrame
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

    rfecv.fit(feature_data, targets)

    # Describe top features
    n_opt = rfecv.named_steps.est.n_features_
    importance = _extract_important_features(
        rfecv.named_steps.dv.get_feature_names(),
        rfecv.named_steps.est.ranking_)
    importance = sort_importances(importance, ascending=True)[:n_opt]

    # Plot RFE accuracy
    rfep = _plot_RFE(rfecv.named_steps.est)

    return rfecv, importance, rfep


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
    scores, predictions, importances, tops = _fit_and_predict_cv(
        X_train, y_train[column], estimator, param_dist, n_jobs, scoring,
        random_state, cv, stratify, calc_feature_importance, parameter_tuning)

    # Print accuracy score to stdout
    print("Estimator Accuracy: {0} ± {1}".format(
        np.mean(scores), np.std(scores)))

    # TODO: save down estimator with tops parameters (currently the estimator
    # would be untrained, and tops parameters are not reported)

    return predictions['prediction'], importances


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
        random_state, parameter_tuning, classification=True)

    # optimize training feature count
    if optimize_feature_selection:
        X_train, X_test, importances = _optimize_feature_selection(
            output_dir=None, X_train=X_train, X_test=None, y_train=y_train,
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
    estimator.fit(X_train, y_train)

    importances = _attempt_to_calculate_feature_importances(
        estimator, calc_feature_importance,
        optimize_feature_selection, importances)

    return estimator, importances


def split_optimize_classify(features, targets, column, estimator,
                            output_dir, test_size=0.2,
                            step=0.05, cv=5, random_state=None, n_jobs=1,
                            optimize_feature_selection=False,
                            parameter_tuning=False, param_dist=None,
                            calc_feature_importance=False, load_data=True,
                            scoring=accuracy_score, classification=True,
                            stratify=True, palette='sirocco',
                            missing_samples='error'):
    # Load, stratify, and split training/test data
    X_train, X_test, y_train, y_test = _prepare_training_data(
        features, targets, column, test_size, random_state,
        load_data=load_data, stratify=stratify,
        missing_samples=missing_samples)

    # optimize training feature count
    if optimize_feature_selection:
        X_train, X_test, importances = _optimize_feature_selection(
            output_dir, X_train, X_test, y_train, estimator, cv, step, n_jobs)
    else:
        importances = None

    # optimize tuning parameters on your training set
    if parameter_tuning:
        # tune parameters
        estimator = _tune_parameters(
            X_train, y_train, estimator, param_dist, n_iter_search=20,
            n_jobs=n_jobs, cv=cv, random_state=random_state).best_estimator_

    # train classifier and predict test set classes
    estimator, accuracy, y_pred = _fit_and_predict(
            X_train, X_test, y_train, y_test, estimator, scoring=scoring)

    # Predict test set values and plot data, as appropriate for estimator type
    predictions, predict_plot = _predict_and_plot(
        output_dir, y_test, y_pred, estimator, accuracy,
        classification=classification, palette=palette)

    importances = _attempt_to_calculate_feature_importances(
            estimator, calc_feature_importance,
            optimize_feature_selection, importances)

    return estimator, predictions, accuracy, importances


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
            features, targets, missing_samples=missing_samples)

    # split into training and test sets
    if stratify:
        strata = targets[column]
    else:
        strata = None

    X_train, X_test, y_train, y_test = _split_training_data(
        features, targets, column, test_size, strata,
        random_state)

    return X_train, X_test, y_train, y_test


def _optimize_feature_selection(output_dir, X_train, X_test, y_train,
                                estimator, cv, step, n_jobs):
    rfecv, importance, rfep = _rfecv_feature_selection(
        X_train, y_train, estimator=estimator, cv=cv, step=step, n_jobs=n_jobs)
    if output_dir:
        rfep.savefig(join(output_dir, 'rfe_plot.png'))
        rfep.savefig(join(output_dir, 'rfe_plot.pdf'))
        plt.close('all')

    index = set(importance.index)
    X_train = [{k: r[k] for k in r.keys() & index} for r in X_train]
    if X_test is not None:
        X_test = [{k: r[k] for k in r.keys() & index} for r in X_test]
    return X_train, X_test, importance


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


def _predict_and_plot(output_dir, y_test, y_pred, estimator, accuracy,
                      classification=True, palette='sirocco'):
    if classification:
        predictions, predict_plot = _plot_confusion_matrix(
            y_test, y_pred, sorted(estimator.classes_), accuracy,
            normalize=True, palette=palette)
    else:
        predictions = _linear_regress(y_test, y_pred)
        predict_plot = _regplot_from_dataframe(y_test, y_pred)
    if output_dir is not None:
        predict_plot.get_figure().savefig(
            join(output_dir, 'predictions.png'), bbox_inches='tight')
        predict_plot.get_figure().savefig(
            join(output_dir, 'predictions.pdf'), bbox_inches='tight')
    return predictions, predict_plot


def sort_importances(importances, ascending=False):
    return importances.sort_values(
        by=importances.columns[0], ascending=ascending)


def _extract_estimator_parameters(estimator):
    # summarize model accuracy and params
    # (drop pipeline params and individual base estimators)
    estimator_params = {k: v for k, v in estimator.get_params().items() if
                        k.startswith('est__') and k != 'est__base_estimator'}
    return pd.Series(estimator_params, name='Parameter setting')


def _visualize(output_dir, estimator, cm, accuracy, importances=None,
               optimize_feature_selection=True, title='results'):

    pd.set_option('display.max_colwidth', -1)

    # summarize model accuracy and params
    result = _extract_estimator_parameters(estimator)
    result = q2templates.df_to_html(result.to_frame())

    cm.to_csv(join(
        output_dir, 'predictive_accuracy.tsv'), sep='\t', index=True)
    cm = q2templates.df_to_html(cm)

    if importances is not None:
        importances = sort_importances(importances)
        pd.set_option('display.float_format', '{:.3e}'.format)
        importances.to_csv(join(
            output_dir, 'feature_importance.tsv'), sep='\t', index=True)
        importances = q2templates.df_to_html(importances, index=True)

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': title,
        'result': result,
        'predictions': cm,
        'importances': importances,
        'classification': True,
        'optimize_feature_selection': optimize_feature_selection,
        'maturity_index': False})


def _visualize_maturity_index(table, metadata, group_by, column,
                              predicted_column, importances, estimator,
                              accuracy, output_dir, maz_stats=True):

    pd.set_option('display.max_colwidth', -1)

    maturity = '{0} maturity'.format(column)
    maz = '{0} MAZ score'.format(column)

    # save feature importance data and convert to html
    importances = sort_importances(importances)
    importances.to_csv(
        join(output_dir, 'feature_importance.tsv'), index=True, sep='\t')
    importance = q2templates.df_to_html(importances, index=True)

    # save predicted values, maturity, and MAZ score data
    maz_md = metadata[[group_by, column, predicted_column, maturity, maz]]
    maz_md.to_csv(join(output_dir, 'maz_scores.tsv'), sep='\t')
    if maz_stats:
        maz_aov = _two_way_anova(table, metadata, maz, group_by, column)[0]
        maz_aov.to_csv(join(output_dir, 'maz_aov.tsv'), sep='\t')
        maz_pairwise = _pairwise_stats(
            table, metadata, maz, group_by, column)
        maz_pairwise.to_csv(join(output_dir, 'maz_pairwise.tsv'), sep='\t')

    # plot control/treatment predicted vs. actual values
    g = _lmplot_from_dataframe(
        metadata, column, predicted_column, group_by)
    g.savefig(join(output_dir, 'maz_predictions.png'), bbox_inches='tight')
    g.savefig(join(output_dir, 'maz_predictions.pdf'), bbox_inches='tight')
    plt.close('all')

    # plot barplots of MAZ score vs. column (e.g., age)
    g = _boxplot_from_dataframe(metadata, column, maz, group_by)
    g.get_figure().savefig(
        join(output_dir, 'maz_boxplots.png'), bbox_inches='tight')
    g.get_figure().savefig(
        join(output_dir, 'maz_boxplots.pdf'), bbox_inches='tight')
    plt.close('all')

    # plot heatmap of column (e.g., age) vs. abundance of top features
    top = table[list(importances.index)]
    g = _clustermap_from_dataframe(top, metadata, group_by, column)
    g.savefig(join(output_dir, 'maz_heatmaps.png'), bbox_inches='tight')
    g.savefig(join(output_dir, 'maz_heatmaps.pdf'), bbox_inches='tight')

    result = _extract_estimator_parameters(estimator)
    result.append(pd.Series([accuracy], index=['Accuracy score']))
    result = q2templates.df_to_html(result.to_frame())

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': 'maturity index predictions',
        'result': result,
        'predictions': None,
        'importances': importance,
        'classification': False,
        'optimize_feature_selection': True,
        'maturity_index': True})


def _map_params_to_pipeline(param_dist):
    return {'est__' + param: dist for param, dist in param_dist.items()}


def _tune_parameters(X_train, y_train, estimator, param_dist, n_iter_search=20,
                     n_jobs=1, cv=None, random_state=None):
    # run randomized search
    random_search = RandomizedSearchCV(
        estimator, param_distributions=param_dist, n_iter=n_iter_search,
        n_jobs=n_jobs, cv=cv, random_state=random_state)
    random_search.fit(X_train, y_train)
    return random_search


def _fit_and_predict(X_train, X_test, y_train, y_test, estimator,
                     scoring=accuracy_score):
    '''train and test estimators.
    scoring: str
        use accuracy_score for classification, mean_squared_error for
        regression.
    '''
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    accuracy = scoring(y_test, pd.DataFrame(y_pred))

    return estimator, accuracy, y_pred


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
            estimator.fit(X_train, y_train)
        # predict values for outer loop test set
        test_set = features[test_index]
        index = metadata.iloc[test_index]
        pred = pd.DataFrame(estimator.predict(test_set), index=index.index)

        # log predictions results
        predictions = pd.concat([predictions, pred])
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

    return scores, predictions, importances, tops


def _mean_feature_importance(importances):
    '''Calculate mean feature importance across a list of pd.dataframes
    containing importance scores of the same features from multiple models
    (e.g., CV importance scores).
    '''
    imp = pd.concat(importances, axis=1)
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


def _maz_score(metadata, predicted, column, group_by, control):
    '''pd.DataFrame -> pd.DataFrame'''
    # extract control data
    md_control = metadata[metadata[group_by] == control]

    # for each bin, calculate median and SD in control samples
    medians = {}
    for n in md_control[column].unique():
        _bin = md_control[md_control[column] == n]
        _median = _bin[predicted].median()
        _std = _bin[predicted].std()
        medians[n] = (_median, _std)

    # calculate maturity and MAZ scores in all samples
    maturity_scores = []
    maz_scores = []
    for i, v in metadata[predicted].iteritems():
        _median, _std = medians[metadata.loc[i][column]]
        maturity = v - _median
        maturity_scores.append(maturity)
        if maturity == 0.0 or _std == 0.0:
            maz_score = 0.0
        else:
            maz_score = maturity / _std
        maz_scores.append(maz_score)

    maturity = '{0} maturity'.format(column)
    metadata[maturity] = maturity_scores
    maz = '{0} MAZ score'.format(column)
    metadata[maz] = maz_scores

    return metadata


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
        estimator = SVR(kernel='rbf')
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
        param_dist = parameters['linear_svm']
        estimator = LinearSVC(random_state=random_state)
    elif estimator == 'SVC':
        param_dist = parameters['svm']
        estimator = SVC(kernel='rbf', random_state=random_state)
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
