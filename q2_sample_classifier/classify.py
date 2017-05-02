#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

import qiime2
import q2templates
import biom
import pandas as pd
from scipy.stats import randint
from os.path import join
import pkg_resources

from .utilities import split_optimize_classify


TEMPLATES = pkg_resources.resource_filename('q2_sample_classifier', 'assets')


def classify_random_forest(output_dir: str, table: biom.Table,
                           metadata: qiime2.Metadata, category: str,
                           test_size: float=0.2, step: float=0.05,
                           cv: int=5, random_state: int=None, n_jobs: int=1,
                           n_estimators: int=10,
                           optimize_feature_selection: bool=False,
                           parameter_tuning: bool=False) -> None:

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {"max_depth": [4, 8, 16, None],
                  "max_features": randint(1, 11),
                  "min_samples_split": [0.001, 0.01, 0.1],
                  "min_weight_fraction_leaf": [0.0001, 0.001, 0.01],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    estimator = RandomForestClassifier(
        n_jobs=n_jobs, n_estimators=n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True)

    # return estimator, cm, accuracy, importances

    importances.to_csv(join(output_dir, 'feature_importance.tsv'), sep='\t')

    pd.set_option('display.max_colwidth', -1)
    result = pd.Series([str(estimator), accuracy],
                       index=['Parameters', 'Accuracy score'],
                       name='Random forest classification results')

    result = result.to_frame().to_html(classes=(
        "table table-striped table-hover")).replace('border="1"', 'border="0"')
    cm = cm.to_html(classes=(
        "table table-striped table-hover")).replace('border="1"', 'border="0"')
    importances = importances.to_html(classes=(
        "table table-striped table-hover")).replace('border="1"', 'border="0"')

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'result': result,
        'predictions': cm,
        'importances': importances,
        'classification': True,
        'optimize_feature_selection': optimize_feature_selection})


def regression_random_forest(table: biom.Table, metadata: qiime2.Metadata,
                             category, test_size=0.2,
                             step=0.05, cv=5, random_state=None, n_jobs=4,
                             n_estimators=10, optimize_feature_selection=False,
                             parameter_tuning=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {"max_depth": [4, 8, 16, None],
                  "max_features": randint(1, 11),
                  "min_samples_split": [0.001, 0.01, 0.1],
                  "min_weight_fraction_leaf": [0.0001, 0.001, 0.01],
                  "bootstrap": [True, False]}

    estimator = RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True, scoring=mean_squared_error,
        classification=False)

    return estimator, cm, accuracy, importances


def classify_linearSVC(table: biom.Table, metadata: qiime2.Metadata, category,
                       test_size=0.2, step=0.05, cv=5,
                       random_state=None, n_jobs=4,
                       optimize_feature_selection=False,
                       parameter_tuning=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {"C": [1, 0.9, 0.8],
                  "penalty": ["l1", "l2"],
                  "loss": ["hinge", "squared_hinge"],
                  "dual": [True, False]}

    estimator = LinearSVC()

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False)

    return estimator, cm, accuracy, importances


# Need to figure out how to pickle/import estimators
def classify_new_data(table: biom.Table, estimator: Pipeline):
    '''Use trained estimator to predict values on unseen data.'''
    predictions = estimator.predict(table)
    return predictions
