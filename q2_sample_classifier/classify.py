#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC, LinearSVR, SVR, SVC
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import qiime2
import biom
from scipy.stats import randint
import warnings

from .utilities import split_optimize_classify, visualize


ensemble_params = {"max_depth": [4, 8, 16, None],
                   "max_features": [None, 'sqrt', 'log2', 0.1],
                   "min_samples_split": [0.001, 0.01, 0.1],
                   "min_weight_fraction_leaf": [0.0001, 0.001, 0.01],
                   "bootstrap": [True, False]}


linear_svm_params = {"C": [1, 0.5, 0.1, 0.9, 0.8],
             # should probably include penalty in grid search, but:
             # Unsupported set of arguments: The combination of
             # penalty='l1' and loss='hinge' is not supported
             # "penalty": ["l1", "l2"],
             "loss": ["hinge", "squared_hinge"],
             "tol": [0.00001, 0.0001, 0.001]
             # should probably include this in grid search, as dual=False
             # is preferred when samples > features. However:
             # Unsupported set of arguments: The combination of
             # penalty='l2' and loss='hinge' are not supported when
             # dual=False
             # "dual": [True, False]
}


svm_params = {"C": [1, 0.5, 0.1, 0.9, 0.8],
              "tol": [0.00001, 0.0001, 0.001, 0.01],
              "shrinking": [True, False],
}


neighbors_params = {
    "n_neighbors": randint(2, 15),
    "weights": ['uniform', 'distance'],
    "leaf_size": randint(15, 100)
}


linear_params = {
    "alpha": [1.0, 1.5, 3.0, 10.0, 100.0],
    "tol": [0.00001, 0.0001, 0.001, 0.01]
}


def param_warning():
    warnings.warn(('This estimator currently does not support parameter '
                   'tuning. Predictions are being made using an un-tuned '
                   'estimator.'), UserWarning)


def classify_random_forest(output_dir: str, table: biom.Table,
                           metadata: qiime2.Metadata, category: str,
                           test_size: float=0.2, step: float=0.05,
                           cv: int=5, random_state: int=None, n_jobs: int=1,
                           n_estimators: int=100,
                           optimize_feature_selection: bool=False,
                           parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {**ensemble_params, "criterion": ["gini", "entropy"]}

    estimator = RandomForestClassifier(
        n_jobs=n_jobs, n_estimators=n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True)

    visualize(output_dir, estimator, cm, accuracy, importances,
              optimize_feature_selection)


def classify_extra_trees(output_dir: str, table: biom.Table,
                         metadata: qiime2.Metadata, category: str,
                         test_size: float=0.2, step: float=0.05,
                         cv: int=5, random_state: int=None, n_jobs: int=1,
                         n_estimators: int=100,
                         optimize_feature_selection: bool=False,
                         parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {**ensemble_params, "criterion": ["gini", "entropy"]}

    estimator = ExtraTreesClassifier(
        n_jobs=n_jobs, n_estimators=n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True)

    visualize(output_dir, estimator, cm, accuracy, importances,
              optimize_feature_selection)


# currently does not support parameter tuning, as the tuning parameters need to
# be fed to the DecisionTreeClassifier estimator, not to the AdaBoostClassifier
# meta-estimator. Currently raises warning if users attempt to tune.
def classify_adaboost(output_dir: str, table: biom.Table,
                      metadata: qiime2.Metadata, category: str,
                      test_size: float=0.2, step: float=0.05,
                      cv: int=5, random_state: int=None, n_jobs: int=1,
                      n_estimators: int=100,
                      optimize_feature_selection: bool=False,
                      parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {k: ensemble_params[k] for k in ensemble_params.keys()
                  if k != "bootstrap"}

    if parameter_tuning:
        param_warning()

    estimator = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=False, param_dist=param_dist,
        calc_feature_importance=True)

    visualize(output_dir, estimator, cm, accuracy, importances,
              optimize_feature_selection)


def classify_gradient_boosting(output_dir: str, table: biom.Table,
                               metadata: qiime2.Metadata, category: str,
                               test_size: float=0.2, step: float=0.05,
                               cv: int=5, random_state: int=None,
                               n_jobs: int=1, n_estimators: int=100,
                               optimize_feature_selection: bool=False,
                               parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {k: ensemble_params[k] for k in ensemble_params.keys()
                  if k != "bootstrap"}

    estimator = GradientBoostingClassifier(n_estimators=n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True)

    visualize(output_dir, estimator, cm, accuracy, importances,
              optimize_feature_selection)


def regress_random_forest(output_dir: str, table: biom.Table,
                          metadata: qiime2.Metadata, category: str,
                          test_size: float=0.2, step: float=0.05,
                          cv: int=5, random_state: int=None, n_jobs: int=1,
                          n_estimators: int=100,
                          optimize_feature_selection: bool=False,
                          parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = ensemble_params

    estimator = RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True, scoring=mean_squared_error,
        classification=False)

    visualize(output_dir, estimator, cm, accuracy, importances,
              optimize_feature_selection)


def regress_extra_trees(output_dir: str, table: biom.Table,
                        metadata: qiime2.Metadata, category: str,
                        test_size: float=0.2, step: float=0.05,
                        cv: int=5, random_state: int=None, n_jobs: int=1,
                        n_estimators: int=100,
                        optimize_feature_selection: bool=False,
                        parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = ensemble_params

    estimator = ExtraTreesRegressor(n_jobs=n_jobs, n_estimators=n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True, scoring=mean_squared_error,
        classification=False)

    visualize(output_dir, estimator, cm, accuracy, importances,
              optimize_feature_selection)


def classify_linearSVC(output_dir: str, table: biom.Table,
                       metadata: qiime2.Metadata, category: str,
                       test_size: float=0.2, step: float=0.05,
                       cv: int=5, random_state: int=None, n_jobs: int=1,
                       parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = linear_svm_params

    estimator = LinearSVC()

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


def classify_SVC(output_dir: str, table: biom.Table,
                 metadata: qiime2.Metadata, category: str,
                 test_size: float=0.2, step: float=0.05,
                 cv: int=5, random_state: int=None, n_jobs: int=1,
                 parameter_tuning: bool=False, kernel: str='rbf'):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = svm_params

    estimator = SVC(kernel=kernel)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


def regress_linearSVR(output_dir: str, table: biom.Table,
                      metadata: qiime2.Metadata, category: str,
                      test_size: float=0.2, step: float=0.05,
                      cv: int=5, random_state: int=None, n_jobs: int=1,
                      parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {**linear_svm_params, 'epsilon': [0.0, 0.1]}

    estimator = LinearSVR()

    # *** Bug: currently parameter tuning fails for SVR only. Error:
    # shapes (25,1107) and (13284,) not aligned: 1107 (dim 1) != 13284 (dim0)
    # *** turning parameter tuning off by default until resolved
    if parameter_tuning:
        param_warning()

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        # parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False, scoring=mean_squared_error,
        classification=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


def regress_SVR(output_dir: str, table: biom.Table,
                metadata: qiime2.Metadata, category: str,
                test_size: float=0.2, step: float=0.05,
                cv: int=5, random_state: int=None, n_jobs: int=1,
                parameter_tuning: bool=False, kernel: str='rbf'):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {**svm_params, 'epsilon': [0.0, 0.1]}

    estimator = SVR(kernel=kernel)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False, scoring=mean_squared_error,
        classification=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


def regress_ridge(output_dir: str, table: biom.Table,
                  metadata: qiime2.Metadata, category: str,
                  test_size: float=0.2, step: float=0.05,
                  cv: int=5, random_state: int=None, n_jobs: int=1,
                  parameter_tuning: bool=False, solver: str='auto'):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = linear_params

    estimator = Ridge(solver=solver)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False, scoring=mean_squared_error,
        classification=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


def regress_lasso(output_dir: str, table: biom.Table,
                  metadata: qiime2.Metadata, category: str,
                  test_size: float=0.2, step: float=0.05,
                  cv: int=5, random_state: int=None, n_jobs: int=1,
                  parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = linear_params

    estimator = Lasso()

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False, scoring=mean_squared_error,
        classification=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


def regress_elasticnet(output_dir: str, table: biom.Table,
                  metadata: qiime2.Metadata, category: str,
                  test_size: float=0.2, step: float=0.05,
                  cv: int=5, random_state: int=None, n_jobs: int=1,
                  parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = linear_params

    estimator = ElasticNet()

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False, scoring=mean_squared_error,
        classification=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


def classify_kneighbors(output_dir: str, table: biom.Table,
                        metadata: qiime2.Metadata, category: str,
                        test_size: float=0.2, step: float=0.05,
                        cv: int=5, random_state: int=None, n_jobs: int=1,
                        parameter_tuning: bool=False, algorithm: str='auto'):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = neighbors_params

    estimator = KNeighborsClassifier(algorithm=algorithm)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


def regress_kneighbors(output_dir: str, table: biom.Table,
                        metadata: qiime2.Metadata, category: str,
                        test_size: float=0.2, step: float=0.05,
                        cv: int=5, random_state: int=None, n_jobs: int=1,
                        parameter_tuning: bool=False, algorithm: str='auto'):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = neighbors_params

    estimator = KNeighborsRegressor(algorithm=algorithm)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=False,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=False, scoring=mean_squared_error,
        classification=False)

    visualize(output_dir, estimator, cm, accuracy, importances)


# Need to figure out how to pickle/import estimators
def classify_new_data(table: biom.Table, estimator: Pipeline):
    '''Use trained estimator to predict values on unseen data.'''
    predictions = estimator.predict(table)
    return predictions
