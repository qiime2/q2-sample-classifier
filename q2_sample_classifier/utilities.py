#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from .visuals import linear_regress, plot_confusion_matrix, plot_RFE


def biom_to_pandas(table):
    """biom.Table->pandas.DataFrame"""
    m = table.matrix_data
    data = [pd.SparseSeries(m[i].toarray().ravel())
            for i in np.arange(m.shape[0])]
    out = pd.SparseDataFrame(data, index=table.ids('observation'),
                             columns=table.ids('sample'))
    return out


def load_data(features_fp, targets_fp, transpose=True):
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
    regr_rf: array
        Feature importance scores or ranking of scores.
    '''
    importances = pd.DataFrame([i for i in zip(table.columns, top)],
                               columns=["feature", "importance"])
    return importances.sort_values(by="importance", ascending=ascending)


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


def split_optimize_classify(features_fp, targets_fp, category, estimator,
                            output_dir, transpose=True, test_size=0.2,
                            step=0.05, cv=5, random_state=None, n_jobs=4,
                            optimize_feature_selection=False,
                            parameter_tuning=False, param_dist=None,
                            calc_feature_importance=False,
                            scoring=accuracy_score, classification=True):
    # load data
    features, targets = load_data(features_fp, targets_fp, transpose=transpose)

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
        predictions, predict_plot = linear_regress(y_test, y_pred, plot=True)
    predict_plot.get_figure().savefig(
        join(output_dir, 'predictions.png'), bbox_inches='tight')
    predict_plot.get_figure().savefig(
        join(output_dir, 'predictions.pdf'), bbox_inches='tight')

    if calc_feature_importance:
        importances = extract_important_features(
            X_train, estimator.feature_importances_)
    else:
        importances = None

    return estimator, predictions, accuracy, importances


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
