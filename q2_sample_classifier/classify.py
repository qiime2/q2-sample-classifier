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
                              AdaBoostRegressor, GradientBoostingRegressor,
                              IsolationForest)
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import qiime2
import biom
import pandas as pd
from scipy.stats import randint
import warnings

from .utilities import (split_optimize_classify, _visualize, _load_data,
                        tune_parameters, _maz_score, _visualize_maturity_index,
                        _split_training_data)
from .visuals import linear_regress


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
                     # should probably include this in grid search, as
                     # dual=False is preferred when samples>features. However:
                     # Unsupported set of arguments: The combination of
                     # penalty='l2' and loss='hinge' are not supported when
                     # dual=False
                     # "dual": [True, False]
                     }


svm_params = {"C": [1, 0.5, 0.1, 0.9, 0.8],
              "tol": [0.00001, 0.0001, 0.001, 0.01],
              "shrinking": [True, False]}


neighbors_params = {
    "n_neighbors": randint(2, 15),
    "weights": ['uniform', 'distance'],
    "leaf_size": randint(15, 100)
}


linear_params = {
    "alpha": [0.0001, 0.01, 1.0, 10.0, 1000.0],
    "tol": [0.00001, 0.0001, 0.001, 0.01]
}


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

    _visualize(output_dir, estimator, cm, accuracy, importances,
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

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def classify_adaboost(output_dir: str, table: biom.Table,
                      metadata: qiime2.Metadata, category: str,
                      test_size: float=0.2, step: float=0.05,
                      cv: int=5, random_state: int=None, n_jobs: int=1,
                      n_estimators: int=100,
                      optimize_feature_selection: bool=False,
                      parameter_tuning: bool=False):

    base_estimator = DecisionTreeClassifier()

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {k: ensemble_params[k] for k in ensemble_params.keys()
                  if k != "bootstrap"}

    # parameter tune base estimator
    if parameter_tuning:
        features, targets = _load_data(table, metadata, transpose=True)
        base_estimator = tune_parameters(
            features, targets[category], base_estimator, param_dist,
            n_jobs=n_jobs, cv=cv, random_state=random_state)

    estimator = AdaBoostClassifier(base_estimator, n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=False, param_dist=param_dist,
        calc_feature_importance=True)

    _visualize(output_dir, estimator, cm, accuracy, importances,
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

    _visualize(output_dir, estimator, cm, accuracy, importances,
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

    _visualize(output_dir, estimator, cm, accuracy, importances,
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

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def regress_adaboost(output_dir: str, table: biom.Table,
                     metadata: qiime2.Metadata, category: str,
                     test_size: float=0.2, step: float=0.05,
                     cv: int=5, random_state: int=None, n_jobs: int=1,
                     n_estimators: int=100,
                     optimize_feature_selection: bool=False,
                     parameter_tuning: bool=False):

    base_estimator = DecisionTreeRegressor()

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {k: ensemble_params[k] for k in ensemble_params.keys()
                  if k != "bootstrap"}

    # parameter tune base estimator
    if parameter_tuning:
        features, targets = _load_data(table, metadata, transpose=True)
        base_estimator = tune_parameters(
            features, targets[category], base_estimator, param_dist,
            n_jobs=n_jobs, cv=cv, random_state=random_state)

    estimator = AdaBoostRegressor(base_estimator, n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=False, param_dist=param_dist,
        calc_feature_importance=True, scoring=mean_squared_error,
        classification=False)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def regress_gradient_boosting(output_dir: str, table: biom.Table,
                              metadata: qiime2.Metadata, category: str,
                              test_size: float=0.2, step: float=0.05,
                              cv: int=5, random_state: int=None,
                              n_jobs: int=1, n_estimators: int=100,
                              optimize_feature_selection: bool=False,
                              parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {k: ensemble_params[k] for k in ensemble_params.keys()
                  if k != "bootstrap"}

    estimator = GradientBoostingRegressor(n_estimators=n_estimators)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True, scoring=mean_squared_error,
        classification=False)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def classify_linearSVC(output_dir: str, table: biom.Table,
                       metadata: qiime2.Metadata, category: str,
                       test_size: float=0.2, step: float=0.05,
                       cv: int=5, random_state: int=None, n_jobs: int=1,
                       parameter_tuning: bool=False,
                       optimize_feature_selection: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = linear_svm_params

    estimator = LinearSVC()

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def classify_SVC(output_dir: str, table: biom.Table,
                 metadata: qiime2.Metadata, category: str,
                 test_size: float=0.2, step: float=0.05,
                 cv: int=5, random_state: int=None, n_jobs: int=1,
                 parameter_tuning: bool=False,
                 optimize_feature_selection: bool=False, kernel: str='rbf'):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = svm_params

    estimator = SVC(kernel=kernel)

    # linear SVC returns feature weights as coef_
    calc_feature_importance, optimize_feature_selection = svm_set(
        kernel, optimize_feature_selection)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=calc_feature_importance)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def regress_SVR(output_dir: str, table: biom.Table,
                metadata: qiime2.Metadata, category: str,
                test_size: float=0.2, step: float=0.05,
                cv: int=5, random_state: int=None, n_jobs: int=1,
                parameter_tuning: bool=False,
                optimize_feature_selection: bool=False, kernel: str='rbf'):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = {**svm_params, 'epsilon': [0.0, 0.1]}

    estimator = SVR(kernel=kernel)

    # linear SVR returns feature weights as coef_ , non-linear does not
    calc_feature_importance, optimize_feature_selection = svm_set(
        kernel, optimize_feature_selection)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=calc_feature_importance,
        scoring=mean_squared_error, classification=False)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def regress_ridge(output_dir: str, table: biom.Table,
                  metadata: qiime2.Metadata, category: str,
                  test_size: float=0.2, step: float=0.05,
                  cv: int=5, random_state: int=None, n_jobs: int=1,
                  parameter_tuning: bool=False,
                  optimize_feature_selection: bool=False, solver: str='auto'):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = linear_params

    estimator = Ridge(solver=solver)

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True, scoring=mean_squared_error,
        classification=False)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def regress_lasso(output_dir: str, table: biom.Table,
                  metadata: qiime2.Metadata, category: str,
                  test_size: float=0.2, step: float=0.05,
                  cv: int=5, random_state: int=None, n_jobs: int=1,
                  optimize_feature_selection: bool=False,
                  parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = linear_params

    estimator = Lasso()

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True, scoring=mean_squared_error,
        classification=False)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


def regress_elasticnet(output_dir: str, table: biom.Table,
                       metadata: qiime2.Metadata, category: str,
                       test_size: float=0.2, step: float=0.05,
                       cv: int=5, random_state: int=None, n_jobs: int=1,
                       optimize_feature_selection: bool=False,
                       parameter_tuning: bool=False):

    # specify parameters and distributions to sample from for parameter tuning
    param_dist = linear_params

    estimator = ElasticNet()

    estimator, cm, accuracy, importances = split_optimize_classify(
        table, metadata, category, estimator, output_dir,
        test_size=test_size, step=step, cv=cv, random_state=random_state,
        n_jobs=n_jobs, optimize_feature_selection=optimize_feature_selection,
        parameter_tuning=parameter_tuning, param_dist=param_dist,
        calc_feature_importance=True, scoring=mean_squared_error,
        classification=False)

    _visualize(output_dir, estimator, cm, accuracy, importances,
               optimize_feature_selection)


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

    _visualize(output_dir, estimator, cm, accuracy, importances, False)


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

    _visualize(output_dir, estimator, cm, accuracy, importances, False)


def maturity_index(output_dir: str, table: biom.Table,
                   metadata: qiime2.Metadata, category: str, group_by: str,
                   control: str, estimator: str='RandomForestRegressor',
                   n_estimators: int=100, test_size: float=0.2,
                   step: float=0.05, cv: int=5, random_state: int=None,
                   n_jobs: int=1, parameter_tuning: bool=True,
                   optimize_feature_selection: bool=True,
                   maz_stats: bool=True):
    '''Calculate a "maturity index" to predict values of a continuous
    metadata category as a function of microbiota composition. A "normal"
    maturation profile is trained based on a set of control samples. MAZ scores
    are then calculated for all samples. Plots predicted vs. expected values
    for each group in category; barplots of MAZ scores for each group in
    category; and heatmap of top feature abundance X category. CATEGORY MUST
    BE PREVIOUSLY BINNED INTO SENSIBLE BINS, E.G., MONTHS INSTEAD OF DAYS.

    category: str
        Continuous metadata category to use for estimator fitting/prediction.
    group_by: str
        Metadata category to use for plotting and significance testing.
    control: str
        Value of group_by to use as control group.
    '''
    # select estimator
    estimator, param_dist = select_estimator(estimator, n_jobs, n_estimators)

    # split input data into control and treatment groups
    table, metadata = _load_data(table, metadata, transpose=True)
    md_control = metadata[metadata[group_by] == control]
    table_control = table.ix[list(md_control.index.values)]

    # train model on control data
    estimator, cm, accuracy, importances = split_optimize_classify(
        table_control, md_control, category, estimator, output_dir,
        random_state=random_state, n_jobs=n_jobs, test_size=test_size,
        step=step, cv=cv, parameter_tuning=parameter_tuning,
        optimize_feature_selection=optimize_feature_selection,
        param_dist=param_dist, calc_feature_importance=True, load_data=False,
        scoring=mean_squared_error, classification=False)

    # predict treatment data
    table = table.loc[:, importances["feature"]]
    y_pred = estimator.predict(table)
    predicted_category = 'predicted {0}'.format(category)
    metadata[predicted_category] = y_pred

    # calculate MAZ score
    metadata = _maz_score(
        metadata, predicted_category, category, group_by, control)

    # visualize
    _visualize_maturity_index(table, metadata, group_by, category,
                              predicted_category, importances, estimator,
                              accuracy, output_dir, maz_stats=maz_stats)


def detect_outliers(table: biom.Table,
                    metadata: qiime2.Metadata, subset_category: str=None,
                    subset_value: str=None, n_estimators: int=100,
                    contamination: float=0.05, random_state: int=None,
                    n_jobs: int=1) -> (pd.Series):
    '''Detect outlier samples within a given sample class. Applications include
    but are not limited to detecting potentially contaminated samples,
    detecting potentially mislabeled samples, and detecting significant
    novelty, e.g., patients who responded to a treatment.

    Input a feature table, possibly filtered to remove samples, depending on
    the goals of this analysis. Outliers can be detected from multiple sample
    types simultaneously, provided the goal is not to detect mislabeled samples
    or samples cross-contaminated with another sample type in this table. E.g.,
    for detecting novelty or exogenous contaminants (e.g., from reagents), many
    different sample types may be tested simultaneously. Otherwise, the feature
    table should be filtered to contain only one or more sample classes between
    which cross-contamination is not suspected, or if these sample classes are
    highly resolved and mislabeled samples are not suspected. These assumptions
    may be supported by a preliminary principal coordinates analysis or other
    diversity analyses to determine how well resolved sample classes are and
    whether some sample classes appear to cluster with the wrong class(es).

    Inputs support two different modes: if subset_category and subset_value are
    set, a subset of the input table is used as a "gold standard" sample pool
    for training the model. This mode is useful, for example, if you have a
    subset of "positive control" samples that represent the known diversity of
    your sample types. Otherwise, the model is trained on all samples.
    Regardless of the input mode used, outlier status is predicted on all
    samples.

    Returns a series of values documenting outlier status: inliers have value
    1, outliers have value -1. This series may be added to a metadata map and
    used to filter a feature table, if appropriate, using
    q2_feature_table.filter_samples, to remove contaminants or focus on novelty
    samples.
    If interested in potentially mislabeled samples, use a sample classifier in
    q2_sample_classifier or principal coordinates analysis to determine whether
    outliers classify as or cluster with another sample type.
    '''
    features, sample_md = _load_data(table, metadata, transpose=True)

    # if opting to train on a subset, choose subset that fits criteria
    if subset_category and subset_value:
        y_train = metadata[metadata[subset_category] == subset_value]
        X_train = table.ix[list(y_train.index.values)]
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
    y_pred.name = "inlier"
    return y_pred


def predict_coordinates(table: biom.Table, metadata: qiime2.Metadata,
                        latitude: str='latitude', longitude: str='longitude',
                        estimator: str='RandomForestRegressor',
                        n_estimators: int=100, test_size: float=0.2,
                        step: float=0.05, cv: int=5, random_state: int=None,
                        n_jobs: int=1, parameter_tuning: bool=True,
                        optimize_feature_selection: bool=True,
                        ) -> (pd.DataFrame, pd.DataFrame):
    '''Predict and map sample coordinates in 2-Dspace, based on
    microbiota composition. E.g., this function could be used to predict
    latitude and longitude or precise location within 2-D physical space,
    such as the built environment. Metadata must be in float format, e.g.,
    decimal degrees geocoordinates.
    '''
    # select estimator
    estimator, param_dist = select_estimator(estimator, n_jobs, n_estimators)

    # split input data into training and test sets
    table, metadata = _load_data(table, metadata, transpose=True)
    X_train, X_test, y_train, y_test = _split_training_data(
        table, metadata, [latitude, longitude], test_size,
        random_state=random_state)

    # train model and predict test data for each category
    # *** would it be better to do this as a multilabel regression?
    # *** currently each dimension is predicted separately
    estimators = {}
    predictions = {}
    prediction_regression = pd.DataFrame()
    for category in [latitude, longitude]:
        estimator, cm, acc, importances = split_optimize_classify(
            X_train, y_train, category, estimator, output_dir=None,
            random_state=random_state, n_jobs=n_jobs, test_size=0.0,
            step=step, cv=cv, parameter_tuning=parameter_tuning,
            optimize_feature_selection=optimize_feature_selection,
            param_dist=param_dist, calc_feature_importance=True,
            load_data=False, scoring=mean_squared_error, classification=False)

        y_pred = estimator.predict(X_test.iloc[:, importances.index])
        predictions[category] = y_pred
        pred = linear_regress(y_test[category], y_pred)
        prediction_regression = pd.concat(
            [prediction_regression, pred.rename(index={0: category})])
        estimators[category] = estimator

    predictions = pd.DataFrame(predictions, index=X_test.index)

    return predictions, prediction_regression


# Need to figure out how to pickle/import estimators
def predict_new_data(table: biom.Table, estimator: Pipeline):
    '''Use trained estimator to predict values on unseen data.'''
    predictions = estimator.predict(table)
    return predictions


def select_estimator(estimator, n_jobs, n_estimators):
    '''Select estimator and parameters from argument name.'''
    if estimator == 'RandomForestRegressor':
        param_dist = ensemble_params
        estimator = RandomForestRegressor(
            n_jobs=n_jobs, n_estimators=n_estimators)
    elif estimator == 'ExtraTreesRegressor':
        param_dist = ensemble_params
        estimator = ExtraTreesRegressor(
            n_jobs=n_jobs, n_estimators=n_estimators)
    elif estimator == 'GradientBoostingRegressor':
        param_dist = {k: ensemble_params[k] for k in ensemble_params.keys()
                      if k != "bootstrap"}
        estimator = GradientBoostingRegressor(n_estimators=n_estimators)
    elif estimator == 'SVR':
        param_dist = {**svm_params, 'epsilon': [0.0, 0.1]}
        estimator = SVR(kernel='linear')
    elif estimator == 'Ridge':
        param_dist = linear_params
        estimator = Ridge(solver='auto')
    elif estimator == 'Lasso':
        param_dist = linear_params
        estimator = Lasso()
    elif estimator == 'ElasticNet':
        param_dist = linear_params
        estimator = ElasticNet()
    return estimator, param_dist


def svm_set(kernel, optimize_feature_selection):
    if kernel == 'linear':
        calc_feature_importance = True
        optimize_feature_selection = optimize_feature_selection
    else:
        calc_feature_importance = False
        optimize_feature_selection = False
        warn_feature_selection()
    return calc_feature_importance, optimize_feature_selection


def warn_feature_selection():
    warning = (
        ('This estimator does not support recursive feature extraction with '
         'the parameter settings requested. See documentation or try a '
         'different estimator model.'))
    warnings.warn(warning, UserWarning)
