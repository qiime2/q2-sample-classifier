#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import q2templates
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import pkg_resources
import warnings
from scipy.stats import randint

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


def _load_data(feature_data, targets_metadata, transpose=False):
    '''Load data and generate training and test sets.

    feature_data: pd.DataFrame
        feature X sample values.
    targets_metadata: qiime2.Metadata
        target (columns) X sample (rows) values.
    transpose: bool
        Transpose feature data? feature tables should be oriented
        to have features (columns) X samples (rows)
    '''
    if transpose is True:
        feature_data = feature_data.transpose()

    # Load metadata, attempt to convert to numeric
    targets = _metadata_to_df(targets_metadata)

    # filter features and targets so samples match
    merged = feature_data.join(targets, how='inner')
    feature_data = feature_data.loc[merged.index]
    targets = targets.loc[merged.index]

    return feature_data, targets


def _metadata_to_df(metadata):
    # Load metadata, attempt to convert to numeric
    metadata = metadata.to_dataframe()
    metadata = metadata.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    return metadata


def _extract_important_features(table, top, ascending=False):
    '''Find top features, match names to indices, sort.
    table: pandas.DataFrame
        Source data table containing samples x features.
    top: array
        Feature importance scores, coef_ scores, or ranking of scores.
    '''
    # is top a 1-d or multi-d array?
    # coef_ is a multidimensional array of shape = [n_class-1, n_features]
    if any(isinstance(i, list) for i in top) or top.ndim > 1:
        # iterate over each list of importances (coef_ sets) in array
        tops = range(len(top))
        imp = pd.DataFrame(
            [i for i in zip(table.columns, *[top[n] for n in tops])],
            columns=["feature", *["importance{0}".format(n) for n in tops]])
    # ensemble estimators and RFECV return 1-d arrays
    else:
        imp = pd.DataFrame([i for i in zip(table.columns, top)],
                           columns=["feature", "importance"])
    return imp.sort_values(by=imp.columns[1], ascending=ascending)


def _split_training_data(feature_data, targets, category, test_size=0.2,
                         stratify=None, random_state=None, drop_na=True):
    '''Split data sets into training and test sets.

    feature_data: pandas.DataFrame
        feature X sample values.
    targets: pandas.DataFrame
        target (columns) X sample (rows) values.
    category: str
        Target category contained in targets.
    test_size: float
        Fraction of data to be reserved as test data.
    stratify: array-like
        Stratify data using this as class labels. E.g., set to df
        category by setting stratify=df[category]
    random_state: int or None
        Int to use for seeding random state. Random if None.
    '''
    # Define target / predictor data
    targets = targets[category]

    if drop_na:
        targets = targets.dropna(axis=0, how='any')
        feature_data = feature_data.loc[targets.index]

    if test_size > 0.0:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                feature_data, targets, test_size=test_size, stratify=stratify,
                random_state=random_state)
        except ValueError:
            raise ValueError((
                'You have chosen to predict a metadata category that contains '
                'one or more values that match only one sample. For proper '
                'stratification of data into training and test sets, each '
                'class (value) must contain at least two samples. This is a '
                'requirement for classification problems, but stratification '
                'can be disabled for regression by setting stratify=False. '
                'Alternatively, remove all samples that bear a unique class '
                'label for your chosen metadata category. Note that disabling '
                'stratification can negatively impact predictive accuracy for '
                'small data sets.'))
    else:
        X_train, X_test, y_train, y_test = (
            feature_data, feature_data, targets, targets)

    return X_train, X_test, y_train, y_test


def _rfecv_feature_selection(feature_data, targets, estimator,
                             cv=5, step=1, scoring=None,
                             random_state=None, n_jobs=4):
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
    top_feature_data: pandas.DataFrame
        feature_data filtered to contain only top features.
    '''

    rfecv = RFECV(estimator=estimator, step=step, cv=cv,
                  scoring=scoring, n_jobs=n_jobs)

    rfecv.fit(feature_data, targets)

    # Describe top features
    n_opt = rfecv.n_features_
    importance = _extract_important_features(
        feature_data, rfecv.ranking_, ascending=True)[:n_opt]
    top_feature_data = feature_data.iloc[:, importance.index]

    # Plot RFE accuracy
    rfep = _plot_RFE(rfecv)

    return rfecv, importance, top_feature_data, rfep


def split_optimize_classify(features, targets, category, estimator,
                            output_dir, transpose=False, test_size=0.2,
                            step=0.05, cv=5, random_state=None, n_jobs=4,
                            optimize_feature_selection=False,
                            parameter_tuning=False, param_dist=None,
                            calc_feature_importance=False, load_data=True,
                            scoring=accuracy_score, classification=True,
                            stratify=True):
    # Load, stratify, and split training/test data
    X_train, X_test, y_train, y_test = _prepare_training_data(
        features, targets, category, test_size, random_state,
        load_data=load_data, transpose=transpose, stratify=stratify)

    # optimize training feature count
    if optimize_feature_selection:
        X_train, X_test, importance = _optimize_feature_selection(
            output_dir, X_train, X_test, y_train, estimator, cv, step,
            random_state, n_jobs)

    # optimize tuning parameters on your training set
    if parameter_tuning:
        # tune parameters
        estimator = _tune_parameters(
            X_train, y_train, estimator, param_dist, n_iter_search=20,
            n_jobs=n_jobs, cv=cv, random_state=random_state)

    # train classifier and predict test set classes
    estimator, accuracy, y_pred = _fit_and_predict(
            X_train, X_test, y_train, y_test, estimator, scoring=scoring)

    # Predict test set values and plot data, as appropriate for estimator type
    predictions, predict_plot = _predict_and_plot(
        output_dir, y_test, y_pred, estimator, accuracy,
        classification=classification)

    # calculate feature importances, if appropriate for the estimator
    if calc_feature_importance:
        importances = _calculate_feature_importances(X_train, estimator)
    # otherwise, if optimizing feature selection, just return ranking from RFE
    elif optimize_feature_selection:
        importances = importance
    # otherwise, we have no weights nor selection, so features==n_features
    else:
        importances = None

    return estimator, predictions, accuracy, importances


def _prepare_training_data(features, targets, category, test_size,
                           random_state, load_data=True, transpose=False,
                           stratify=True):
    # load data
    if load_data:
        features, targets = _load_data(features, targets, transpose=transpose)

    # split into training and test sets
    if stratify:
        strata = targets[category]
    else:
        strata = None

    X_train, X_test, y_train, y_test = _split_training_data(
        features, targets, category, test_size, strata,
        random_state)

    return X_train, X_test, y_train, y_test


def _optimize_feature_selection(output_dir, X_train, X_test, y_train,
                                estimator, cv, step, random_state, n_jobs):
    rfecv, importance, top_feature_data, rfep = _rfecv_feature_selection(
        X_train, y_train, estimator=estimator, cv=cv, step=step,
        random_state=random_state, n_jobs=n_jobs)
    if output_dir:
        rfep.savefig(join(output_dir, 'rfe_plot.png'))
        rfep.savefig(join(output_dir, 'rfe_plot.pdf'))
    plt.close('all')

    X_train = X_train.loc[:, importance["feature"]]
    X_test = X_test.loc[:, importance["feature"]]

    return X_train, X_test, importance


def _calculate_feature_importances(X_train, estimator):
    # only set calc_feature_importance=True if estimator has attributes
    # feature_importances_ or coef_ to report feature importance/weights
    try:
        importances = _extract_important_features(
            X_train, estimator.feature_importances_)
    # is there a better way to determine whether estimator has coef_ ?
    except AttributeError:
        importances = _extract_important_features(
            X_train, estimator.coef_)
    return importances


def _predict_and_plot(output_dir, y_test, y_pred, estimator, accuracy,
                      classification=True):
    if classification:
        predictions, predict_plot = _plot_confusion_matrix(
            y_test, y_pred, sorted(estimator.classes_), accuracy)
    else:
        predictions = _linear_regress(y_test, y_pred)
        predict_plot = _regplot_from_dataframe(y_test, y_pred)
    if output_dir is not None:
        predict_plot.get_figure().savefig(
            join(output_dir, 'predictions.png'), bbox_inches='tight')
        predict_plot.get_figure().savefig(
            join(output_dir, 'predictions.pdf'), bbox_inches='tight')
    return predictions, predict_plot


def _visualize(output_dir, estimator, cm, accuracy, importances=None,
               optimize_feature_selection=True):

    # Need to sort out how to save estimator as sklearn.pipeline
    # This will be possible once qiime2 support pipeline actions

    pd.set_option('display.max_colwidth', -1)

    # summarize model accuracy and params
    result = estimator.get_params()
    result = pd.Series(estimator.get_params(), name='Parameter setting')

    result = result.to_frame().to_html(classes=(
        "table table-striped table-hover")).replace('border="1"', 'border="0"')
    cm = cm.to_html(classes=(
        "table table-striped table-hover")).replace('border="1"', 'border="0"')
    if importances is not None:
        pd.set_option('display.float_format', '{:.3e}'.format)
        importances.to_csv(join(
            output_dir, 'feature_importance.tsv'), sep='\t')
        importances = importances.to_html(classes=(
            "table table-striped table-hover")).replace(
                'border="1"', 'border="0"')

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'result': result,
        'predictions': cm,
        'importances': importances,
        'classification': True,
        'optimize_feature_selection': optimize_feature_selection,
        'maturity_index': False})


def _visualize_maturity_index(table, metadata, group_by, category,
                              predicted_category, importances, estimator,
                              accuracy, output_dir, maz_stats=True):

    pd.set_option('display.max_colwidth', -1)

    maturity = '{0} maturity'.format(category)
    maz = '{0} MAZ score'.format(category)

    # save feature importance data and convert to html
    importances.to_csv(join(output_dir, 'feature_importance.tsv'), sep='\t')
    importance = importances.to_html(classes=(
        "table table-striped table-hover")).replace('border="1"', 'border="0"')

    # save predicted values, maturity, and MAZ score data
    maz_md = metadata[[group_by, category, predicted_category, maturity, maz]]
    maz_md.to_csv(join(output_dir, 'maz_scores.tsv'), sep='\t')
    if maz_stats:
        maz_aov = _two_way_anova(table, metadata, maz, group_by, category)[0]
        maz_aov.to_csv(join(output_dir, 'maz_aov.tsv'), sep='\t')
        maz_pairwise = _pairwise_stats(
            table, metadata, maz, group_by, category)
        maz_pairwise.to_csv(join(output_dir, 'maz_pairwise.tsv'), sep='\t')

    # plot control/treatment predicted vs. actual values
    g = _lmplot_from_dataframe(
        metadata, category, predicted_category, group_by)
    g.savefig(join(output_dir, 'maz_predictions.png'), bbox_inches='tight')
    g.savefig(join(output_dir, 'maz_predictions.pdf'), bbox_inches='tight')
    plt.close('all')

    # plot barplots of MAZ score vs. category (e.g., age)
    g = _boxplot_from_dataframe(metadata, category, maz, group_by)
    g.get_figure().savefig(
        join(output_dir, 'maz_boxplots.png'), bbox_inches='tight')
    g.get_figure().savefig(
        join(output_dir, 'maz_boxplots.pdf'), bbox_inches='tight')
    plt.close('all')

    # plot heatmap of category (e.g., age) vs. abundance of top features
    top = table[list(importances.feature)]
    g = _clustermap_from_dataframe(top, metadata, group_by, category)
    g.savefig(join(output_dir, 'maz_heatmaps.png'), bbox_inches='tight')
    g.savefig(join(output_dir, 'maz_heatmaps.pdf'), bbox_inches='tight')

    result = pd.Series([str(estimator), accuracy],
                       index=['Parameters', 'Accuracy score'],
                       name='Random forest classification results')

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'result': result,
        'predictions': None,
        'importances': importance,
        'classification': False,
        'optimize_feature_selection': True,
        'maturity_index': True})


def _tune_parameters(X_train, y_train, estimator, param_dist, n_iter_search=20,
                     n_jobs=-1, cv=None, random_state=None):
    # run randomized search
    random_search = RandomizedSearchCV(
        estimator, param_distributions=param_dist, n_iter=n_iter_search,
        n_jobs=n_jobs, cv=cv, random_state=random_state)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_


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


def _maz_score(metadata, predicted, category, group_by, control):
    '''pd.DataFrame -> pd.DataFrame'''
    # extract control data
    md_control = metadata[metadata[group_by] == control]

    # for each bin, calculate median and SD in control samples
    medians = {}
    for n in md_control[category].unique():
        _bin = md_control[md_control[category] == n]
        _median = _bin[predicted].median()
        _std = _bin[predicted].std()
        medians[n] = (_median, _std)

    # calculate maturity and MAZ scores in all samples
    maturity_scores = []
    maz_scores = []
    for i, v in metadata[predicted].iteritems():
        _median, _std = medians[metadata.loc[i][category]]
        maturity = v - _median
        maturity_scores.append(maturity)
        if maturity == 0.0 or _std == 0.0:
            maz_score = 0.0
        else:
            maz_score = maturity / _std
        maz_scores.append(maz_score)

    maturity = '{0} maturity'.format(category)
    metadata[maturity] = maturity_scores
    maz = '{0} MAZ score'.format(category)
    metadata[maz] = maz_scores

    return metadata


def _select_estimator(estimator, n_jobs, n_estimators):
    '''Select estimator and parameters from argument name.'''
    # Regressors
    if estimator == 'RandomForestRegressor':
        param_dist = {**parameters['ensemble'], **parameters['bootstrap']}
        estimator = RandomForestRegressor(
            n_jobs=n_jobs, n_estimators=n_estimators)
    elif estimator == 'ExtraTreesRegressor':
        param_dist = {**parameters['ensemble'], **parameters['bootstrap']}
        estimator = ExtraTreesRegressor(
            n_jobs=n_jobs, n_estimators=n_estimators)
    elif estimator == 'GradientBoostingRegressor':
        param_dist = parameters['ensemble']
        estimator = GradientBoostingRegressor(n_estimators=n_estimators)
    elif estimator == 'SVR':
        param_dist = {**parameters['svm'], 'epsilon': [0.0, 0.1]}
        estimator = SVR(kernel='rbf')
    elif estimator == 'LinearSVR':
        param_dist = {**parameters['svm'], 'epsilon': [0.0, 0.1]}
        estimator = SVR(kernel='linear')
    elif estimator == 'Ridge':
        param_dist = parameters['linear']
        estimator = Ridge(solver='auto')
    elif estimator == 'Lasso':
        param_dist = parameters['linear']
        estimator = Lasso()
    elif estimator == 'ElasticNet':
        param_dist = parameters['linear']
        estimator = ElasticNet()
    elif estimator == 'KNeighborsRegressor':
        param_dist = parameters['kneighbors']
        estimator = KNeighborsRegressor(algorithm='auto')

    # Classifiers
    elif estimator == 'RandomForestClassifier':
        param_dist = {**parameters['ensemble'], **parameters['bootstrap'],
                      **parameters['criterion']}
        estimator = RandomForestClassifier(
            n_jobs=n_jobs, n_estimators=n_estimators)
    elif estimator == 'ExtraTreesClassifier':
        param_dist = {**parameters['ensemble'], **parameters['bootstrap'],
                      **parameters['criterion']}
        estimator = ExtraTreesClassifier(
            n_jobs=n_jobs, n_estimators=n_estimators)
    elif estimator == 'GradientBoostingClassifier':
        param_dist = parameters['ensemble']
        estimator = GradientBoostingClassifier(n_estimators=n_estimators)
    elif estimator == 'LinearSVC':
        param_dist = parameters['linear_svm']
        estimator = LinearSVC()
    elif estimator == 'SVC':
        param_dist = parameters['svm']
        estimator = SVC(kernel='rbf')
    elif estimator == 'KNeighborsClassifier':
        param_dist = parameters['kneighbors']
        estimator = KNeighborsClassifier(algorithm='auto')

    return param_dist, estimator


def _train_adaboost_base_estimator(table, metadata, category, n_estimators,
                                   n_jobs, cv, random_state, parameter_tuning,
                                   classification=True):
    param_dist = parameters['ensemble']
    if classification:
        base_estimator = DecisionTreeClassifier()
        adaboost_estimator = AdaBoostClassifier
    else:
        base_estimator = DecisionTreeRegressor()
        adaboost_estimator = AdaBoostRegressor

    if parameter_tuning:
        features, targets = _load_data(table, metadata, transpose=False)
        base_estimator = _tune_parameters(
            features, targets[category], base_estimator, param_dist,
            n_jobs=n_jobs, cv=cv, random_state=random_state)

    return adaboost_estimator(base_estimator, n_estimators)


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


def _set_parameters_and_estimator(estimator, table, metadata, category,
                                  n_estimators, n_jobs, cv, random_state,
                                  parameter_tuning, classification=True):
    # specify parameters and distributions to sample from for parameter tuning
    if estimator in ['AdaBoostClassifier', 'AdaBoostRegressor']:
        estimator = _train_adaboost_base_estimator(
            table, metadata, category, n_estimators, n_jobs, cv, random_state,
            parameter_tuning, classification=classification)
        parameter_tuning = False
        param_dist = None
    else:
        param_dist, estimator = _select_estimator(
            estimator, n_jobs, n_estimators)
    return estimator, param_dist, parameter_tuning


def _warn_feature_selection():
    warning = (
        ('This estimator does not support recursive feature extraction with '
         'the parameter settings requested. See documentation or try a '
         'different estimator model.'))
    warnings.warn(warning, UserWarning)
