#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from sklearn.metrics import mean_squared_error, confusion_matrix

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import linregress
import matplotlib.pyplot as plt


def regplot_from_dataframe(x, y, plot_style="whitegrid", arb=True,
                           color="grey"):
    '''Seaborn regplot with true 1:1 ratio set by arb (bool).'''
    sns.set_style(plot_style)
    reg = sns.regplot(x, y, color=color)
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    if arb is True:
        x0, x1 = reg.axes.get_xlim()
        y0, y1 = reg.axes.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        reg.axes.plot(lims, lims, ':k')
    return reg


def linear_regress(actual, pred, plot=False, plot_style="whitegrid"):
    '''Calculate linear regression on predicted versus expected values.
    actual: pandas.DataFrame
        Actual y-values for test samples.
    pred: pandas.DataFrame
        Predicted y-values for test samples.
    plot: bool
        If True, print seaborn.regplot
    plot_style: str
        Seaborn plot style theme.
    '''
    if plot is True:
        reg = regplot_from_dataframe(actual, pred, plot_style)

    slope, intercept, r_value, p_value, std_err = linregress(actual, pred)
    mse = mean_squared_error(actual, pred)
    return pd.DataFrame([(mse, r_value, p_value, std_err, slope, intercept)],
                        columns=["MSE", "R", "P-val", "Std Error", "Slope",
                                 "Intercept"]), reg


def plot_confusion_matrix(y_test, y_pred, classes, normalize=True):
    cm = confusion_matrix(y_test, y_pred)
    # normalize
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    confusion = sns.heatmap(cm)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, sorted(classes, reverse=True), rotation=0)

    return pd.DataFrame(cm, index=classes, columns=classes), confusion


def plot_RFE(rfecv):
    # If using fractional step, step = integer of fraction * n_features
    if rfecv.step < 1:
        rfecv.step = int(rfecv.step * len(rfecv.ranking_))
    # Need to manually calculate x-axis, as rfecv.grid_scores_ are a 1-d array
    x = [len(rfecv.ranking_) - (n * rfecv.step)
         for n in range(len(rfecv.grid_scores_)-1, -1, -1)]
    if x[0] < 1:
        x[0] = 1

    rfe = plt.figure()
    plt.xlabel("Feature Count")
    plt.ylabel("Accuracy")
    plt.plot(x, rfecv.grid_scores_, 'grey')
    return rfe
