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

import q2templates
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pkg_resources

from .visuals import (linear_regress, plot_confusion_matrix, plot_RFE,
                      pairwise_tests, two_way_anova, clustermap_from_dataframe,
                      boxplot_from_dataframe, lmplot_from_dataframe,
                      regplot_from_dataframe)


TEMPLATES = pkg_resources.resource_filename('q2_sample_classifier', 'assets')


def biom_to_pandas(table):
    """biom.Table->pandas.DataFrame"""
    m = table.matrix_data
    data = [pd.SparseSeries(m[i].toarray().ravel())
            for i in np.arange(m.shape[0])]
    out = pd.SparseDataFrame(data, index=table.ids('observation'),
                             columns=table.ids('sample'))
    return out


def _load_data(features_fp, targets_fp, transpose=True):
    '''Load data and generate training and test sets.

    features_fp: path
        feature X sample values. Currently accepts biom tables.
    targets: path
        target (columns) X sample (rows) values. Currently accepts .tsv
    transpose: bool
        Transpose feature data? biom tables need to be transposed
        to have features (columns) X samples (rows)
    '''
    # convert to df
    feature_data = biom_to_pandas(features_fp)
    if transpose is True:
        feature_data = feature_data.transpose()

    # Load metadata, attempt to convert to numeric
    targets = targets_fp.to_dataframe()
    targets = targets.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    return feature_data, targets


def extract_important_features(table, top, ascending=False):
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


def split_training_data(feature_data, targets, category, test_size=0.2,
                        stratify=None, random_state=None):
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

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, targets, test_size=test_size, stratify=stratify,
        random_state=random_state)

    return X_train, X_test, y_train, y_test


def rfecv_feature_selection(feature_data, targets, estimator,
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
    print("Optimal number of features : {0}".format(n_opt))
    importance = extract_important_features(
        feature_data, rfecv.ranking_, ascending=True)[:n_opt]
    top_feature_data = feature_data.iloc[:, importance.index]

    # Plot RFE accuracy
    rfep = plot_RFE(rfecv)

    return rfecv, importance, top_feature_data, rfep


def split_optimize_classify(features, targets, category, estimator,
                            output_dir, transpose=True, test_size=0.2,
                            step=0.05, cv=5, random_state=None, n_jobs=4,
                            optimize_feature_selection=False,
                            parameter_tuning=False, param_dist=None,
                            calc_feature_importance=False, load_data=True,
                            scoring=accuracy_score, classification=True):
    # load data
    if load_data:
        features, targets = _load_data(features, targets, transpose=transpose)

    # split into training and test sets
    X_train, X_test, y_train, y_test = split_training_data(
        features, targets, category, test_size, targets[category],
        random_state)

    # optimize training feature count
    if optimize_feature_selection:
        rfecv, importance, top_feature_data, rfep = rfecv_feature_selection(
            X_train, y_train, estimator=estimator, cv=cv, step=step,
            random_state=random_state, n_jobs=n_jobs)
        rfep.savefig(join(output_dir, 'rfe_plot.png'))
        rfep.savefig(join(output_dir, 'rfe_plot.pdf'))
        plt.close()

        X_train = X_train.loc[:, importance["feature"]]
        X_test = X_test.loc[:, importance["feature"]]

    # optimize tuning parameters on your training set
    if parameter_tuning:
        # tune parameters
        estimator = tune_parameters(
            X_train, y_train, estimator, param_dist, n_iter_search=20,
            n_jobs=n_jobs, cv=cv, random_state=random_state)

    # train classifier and predict test set classes
    estimator, accuracy, y_pred = fit_and_predict(
            X_train, X_test, y_train, y_test, estimator, scoring=scoring)

    if classification:
        predictions, predict_plot = plot_confusion_matrix(
            y_test, y_pred, sorted(estimator.classes_))
    else:
        predictions = linear_regress(y_test, y_pred)
        predict_plot = regplot_from_dataframe(y_test, y_pred)
    predict_plot.get_figure().savefig(
        join(output_dir, 'predictions.png'), bbox_inches='tight')
    predict_plot.get_figure().savefig(
        join(output_dir, 'predictions.pdf'), bbox_inches='tight')

    # only set calc_feature_importance=True if estimator has attributes
    # feature_importances_ or coef_ to report feature importance/weights
    if calc_feature_importance:
        try:
            importances = extract_important_features(
                X_train, estimator.feature_importances_)
        # is there a better way to determine whether estimator has coef_ ?
        except AttributeError:
            importances = extract_important_features(
                X_train, estimator.coef_)
    # otherwise, if optimizing feature selection, just return ranking from RFE
    elif optimize_feature_selection:
        importances = importance
    # otherwise, we have no weights nor selection, so features==n_features
    else:
        importances = None

    return estimator, predictions, accuracy, importances


def _visualize(output_dir, estimator, cm, accuracy, importances=None,
               optimize_feature_selection=True):

    # Need to sort out how to save estimator as sklearn.pipeline
    # This will be possible once qiime2 support pipeline actions

    pd.set_option('display.max_colwidth', -1)
    result = pd.Series([str(estimator), accuracy],
                       index=['Parameters', 'Accuracy score'],
                       name='Random forest classification results')

    result = result.to_frame().to_html(classes=(
        "table table-striped table-hover")).replace('border="1"', 'border="0"')
    cm = cm.to_html(classes=(
        "table table-striped table-hover")).replace('border="1"', 'border="0"')
    if importances is not None:
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
        maz_aov = two_way_anova(table, metadata, maz, group_by, category)[0]
        maz_aov.to_csv(join(output_dir, 'maz_aov.tsv'), sep='\t')
        maz_pairwise = pairwise_tests(table, metadata, maz, group_by, category)
        maz_pairwise.to_csv(join(output_dir, 'maz_pairwise.tsv'), sep='\t')

    # plot control/treatment predicted vs. actual values
    g = lmplot_from_dataframe(metadata, category, predicted_category, group_by)
    g.savefig(join(output_dir, 'maz_predictions.png'), bbox_inches='tight')
    g.savefig(join(output_dir, 'maz_predictions.pdf'), bbox_inches='tight')
    plt.close()

    # plot barplots of MAZ score vs. category (e.g., age)
    g = boxplot_from_dataframe(metadata, category, maz, group_by)
    g.get_figure().savefig(
        join(output_dir, 'maz_boxplots.png'), bbox_inches='tight')
    g.get_figure().savefig(
        join(output_dir, 'maz_boxplots.pdf'), bbox_inches='tight')
    plt.close()

    # plot heatmap of category (e.g., age) vs. abundance of top features
    top = table[list(importances.feature)]
    g = clustermap_from_dataframe(top, metadata, group_by, category)
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


def tune_parameters(X_train, y_train, estimator, param_dist, n_iter_search=20,
                    n_jobs=-1, cv=None, random_state=None):
    # run randomized search
    random_search = RandomizedSearchCV(
        estimator, param_distributions=param_dist, n_iter=n_iter_search)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_


def fit_and_predict(X_train, X_test, y_train, y_test, estimator,
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
